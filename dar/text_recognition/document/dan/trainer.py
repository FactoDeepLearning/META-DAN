from dar.text_recognition.ocr_manager import OCRManager
from dar.text_recognition.document.dan.attention_visualization import generate_dan_visualization
from dar.generic.utils_visualization import format_attention_weights
from dar.generic.utils import image_torch_to_numpy, rand, randint
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from torch import autocast
import time


class DANManager(OCRManager):
    def __init__(self, params, load_dataset=True):
        super().__init__(params, load_dataset)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        if "add_loop_teacher_forcing" in self.params["training_params"] and self.params["training_params"]["add_loop_teacher_forcing"]:
            if rand() < 0.05:
                y, y_error, y_len = self.inject_loop_error(y, y_error, y_len)
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.tokenizer.tokens["<pad>"]:
                    y_error[b][i] = np.random.randint(0, self.tokenizer.num_tokens)
        return y_error, y, y_len
    
    def inject_loop_error(self, y, y_error, y_len, max_loop_length=20):
        min_length = min(y_len)
        if min_length < 2*(max_loop_length+1):
            return y, y_error, y_len
        loop_length = randint(1, max_loop_length)
        start_index = randint(loop_length+1, min_length-loop_length)
        repeated_portion = y[:, start_index-loop_length:start_index]
        y_len = [l+loop_length for l in y_len]
        y_error = torch.cat([y[:, :start_index], repeated_portion, y[:, start_index:]], dim=1)
        y = torch.cat([y[:, :start_index], y[:, start_index:start_index+loop_length], y[:, start_index:]], dim=1)
        assert y.shape == y_error.shape
        return y, y_error, y_len



    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.tokenizer.tokens["<pad>"])
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        # add errors in teacher forcing
        if "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            cache = None
            raw_features = self.models["encoder"](x)
            features_size = raw_features.size()
            b, c, h, w = features_size

            features = self.models["decoder"].get_pos_features(raw_features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, cache, weights = self.models["decoder"](features,
                                                                               simulated_y_pred[:, :-1],
                                                                               reduced_size,
                                                                               y_len,
                                                                               features_size,
                                                                               start=0,
                                                                               cache=cache,
                                                                               keep_all_weights=True)

            loss_ce = loss_func(pred, y[:, 1:])
            global_loss = loss_ce
            values = {}

            if "add_weights_overlap_penalty" in self.params["training_params"] and self.params["training_params"]["add_weights_overlap_penalty"]:
                for bi in range(b):
                    b_len = y_len[bi]
                    b_weights = weights["mix"][-1][bi][:b_len]
                    # mask_max = b_weights == torch.max(b_weights, dim=0).values
                    # to_minimize = b_weights[~mask_max]
                    # penalty = -torch.sum(torch.log(1-to_minimize))

                    indices = torch.max(b_weights, dim=1).indices
                    to_minimize = torch.clamp(torch.stack([torch.stack([b_weights[k, indices[i]] for k in range(b_weights.size(0)) if k!=i], dim=0) for i in range(b_weights.size(0))]), min=0, max=0.999999)
                    penalty = -torch.sum(torch.log(1-to_minimize))

                    if bi == 0:
                        loss_penalty = penalty/b_len
                    else: 
                        loss_penalty += penalty/b_len
                loss_penalty /= b
                global_loss += self.params["training_params"]["weights_penalty_factor"]*loss_penalty
                values.update({
                    "loss_weights": loss_penalty.item()
                })

            with autocast(device_type=self.autocast_mode, enabled=False):
                self.backward_loss(global_loss)
                self.step_optimizers()
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]

        values.update({
            "nb_samples": b,
            "str_y": batch_data["str_labels"],
            "str_x": str_x,
            "loss": loss_ce.item(),
        })

        values = self.add_syn_values(values)

        return values

    def inference_batch(self, batch_data, return_weights=False):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b,), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.tokenizer.tokens["<start>"]
            predicted_tokens_len = torch.ones((b,), dtype=torch.int, device=self.device)

            all_weights = list()
            whole_output = list()
            confidence_scores = list()
            cache = None

            features = self.models["encoder"](x)
            features_size = features.size()
            features = self.models["decoder"].get_pos_features(features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(0, max_chars):
                output, pred, cache, weights = self.models["decoder"](features, predicted_tokens,
                                                                                      reduced_size,
                                                                                      predicted_tokens_len,
                                                                                      features_size, start=0,
                                                                                      cache=cache, num_pred=1)

                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                reached_end = torch.logical_or(reached_end,
                                               torch.eq(predicted_tokens[:, -1], self.tokenizer.tokens["<end>"]))
                predicted_tokens_len += 1

                prediction_len[reached_end == False] = i + 1
                if torch.all(reached_end):
                    break
                if return_weights:
                    all_weights.append(weights.detach().cpu())

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, 1:]
            prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]].cpu().numpy() for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]
        if return_weights:
            return str_x, confidence_scores, all_weights
        return str_x, confidence_scores

    def evaluate_batch(self, batch_data, metric_names):
        start_time = time.time()
        str_x, confidence_scores, = self.inference_batch(batch_data)

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
        str_x, _, attention_weights = self.inference_batch(batch_data, return_weights=True)

        if output_attention_path:
            img = image_torch_to_numpy(batch_data["imgs"][0])
            attention_weights = [a[0].numpy() for a in attention_weights]
            attention_weights = format_attention_weights(attention_weights, img.shape[:2])
            attention_weights = np.concatenate(attention_weights, axis=0)
            generate_dan_visualization(img, attention_weights, str_x[0], output_attention_path)

        return str_x