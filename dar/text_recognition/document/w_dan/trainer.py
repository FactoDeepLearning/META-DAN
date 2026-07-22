from dar.text_recognition.ocr_manager import OCRManager
from dar.generic.utils import image_torch_to_numpy
from dar.generic.utils_visualization import format_attention_weights
from dar.text_recognition.document.w_dan.attention_visualization import generate_w_dan_visualization
from torch import autocast
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import time


class WDANManager(OCRManager):

    def __init__(self, params, load_dataset=True):
        super().__init__(params, load_dataset)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        token_win = self.params["model_params"]["token_win"]
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(token_win, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.tokenizer.tokens["<pad>"]:
                    y_error[b][i] = np.random.randint(0, self.tokenizer.num_tokens)
        return y_error, y_len

    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.tokenizer.tokens["<pad>"])
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]
        token_win = self.params["model_params"]["token_win"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            cache = None
            raw_features = self.models["encoder"](x)
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].get_pos_features(raw_features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            use_query_emb = self.params["model_params"]["use_query_embedding"]
            window_index = query_index = None
            if use_query_emb:
                num_iter = y.size(1) // token_win - 1
                query_index = torch.arange(token_win, device=self.device).unsqueeze(0).repeat(b, num_iter)
                window_index = torch.arange(num_iter, device=self.device).repeat_interleave(repeats=token_win).unsqueeze(0).repeat(b, 1)

            output, pred, cache, weights = self.models["decoder"](features,
                                                                  simulated_y_pred[:, :-token_win],
                                                                  reduced_size,
                                                                  [max(y_len) for _ in range(b)],
                                                                  features_size,
                                                                  start=0,
                                                                  cache=cache,
                                                                  keep_all_weights=True,
                                                                  query_index=query_index,
                                                                  window_index=window_index)

            loss_ce = loss_func(pred, y[:, token_win:])
            with autocast(device_type=self.autocast_mode, enabled=False):
                self.backward_loss(loss_ce)
                self.step_optimizers()
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": batch_data["str_labels"],
            "str_x": str_x,
            "loss": loss_ce.item(),
        }

        values = self.add_syn_values(values)

        return values

    def inference_batch(self, batch_data, return_weights=False):
        x = batch_data["imgs"].to(self.device)
        token_win = self.params["model_params"]["token_win"]
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        max_chars = self.params["training_params"]["max_char_prediction"]
        use_query_emb = self.params["model_params"]["use_query_embedding"]
        window_index = query_index = None

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, token_win), dtype=torch.long, device=self.device) * self.tokenizer.tokens["<start>"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device) * token_win
            if use_query_emb:
                query_index = torch.arange(token_win, device=self.device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
                window_index = torch.zeros(token_win, device=self.device, dtype=torch.long).unsqueeze(0).repeat(b, 1)

            all_weights = list()
            confidence_scores = list()
            cache = None

            features = self.models["encoder"](x)
            features_size = features.size()
            pos_features = self.models["decoder"].get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(0, max_chars//token_win):
                output, pred, cache, weights = self.models["decoder"](features,
                                                                      predicted_tokens,
                                                                      reduced_size,
                                                                      predicted_tokens_len,
                                                                      features_size,
                                                                      start=0,
                                                                      cache=cache,
                                                                      num_pred=token_win,
                                                                      window_index=window_index,
                                                                      query_index=query_index)
                if return_weights:
                    all_weights.append(weights.detach().cpu())
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -token_win:], dim=1, keepdim=False)], dim=1)
                new_ending = torch.sum(torch.eq(predicted_tokens[:, -token_win:], self.tokenizer.tokens["<end>"]), dtype=bool, dim=-1, keepdim=True)
                reached_end = torch.logical_or(reached_end, new_ending)
                predicted_tokens_len += token_win

                # prediction_len[reached_end == False] = i + 1
                if torch.all(reached_end):
                    break

                if use_query_emb:
                    window_index = torch.cat([window_index, torch.full((b, token_win), fill_value=i+1, device=self.device, dtype=torch.long)], dim=1)
                    query_index = torch.cat([query_index, torch.arange(token_win, device=self.device, dtype=torch.long).unsqueeze(0).repeat(b, 1)], dim=1)

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, token_win:]
            # prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            for i in range(b):
                end_token_indices = torch.nonzero(predicted_tokens[i] == self.tokenizer.tokens["<end>"])
                prediction_len[i] = max_chars - 1 if end_token_indices.numel() == 0 else end_token_indices[0]
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]].cpu().numpy() for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]
        if return_weights:
            return str_x, confidence_scores, all_weights
        return str_x, confidence_scores

    def evaluate_batch(self, batch_data, metric_names):
        start_time = time.time()
        str_x, confidence_scores = self.inference_batch(batch_data)
        process_time = time.time() - start_time

        values = {
            "nb_samples": len(batch_data["imgs"]),
            "str_y": batch_data["str_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values

    def predict_batch(self, batch_data, output_attention_path=None):
        if output_attention_path:
            str_x, _, attention_weights = self.inference_batch(batch_data, return_weights=True)
            img = image_torch_to_numpy(batch_data["imgs"][0])
            attention_weights = format_attention_weights([a[0].numpy() for a in attention_weights], img.shape[:2])
            attention_weights = np.concatenate([np.expand_dims(a, axis=0) for a in attention_weights], axis=0)
            generate_w_dan_visualization(img, attention_weights, str_x[0], output_attention_path)
        else:
            str_x, _ = self.inference_batch(batch_data)
        return str_x