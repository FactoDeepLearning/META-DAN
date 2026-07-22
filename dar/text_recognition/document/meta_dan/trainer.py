from dar.text_recognition.ocr_manager import OCRManager
from dar.generic.utils import image_torch_to_numpy
from dar.text_recognition.document.meta_dan.attention_visualization import generate_meta_dan_visualization
from dar.generic.utils_visualization import format_attention_weights
from torch import autocast
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import time
import os


class MetaDANManager(OCRManager):

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
        num_token_heads = self.params["model_params"]["num_token_heads"]

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



            output, pred, cache, weights = self.models["decoder"](features,
                                                                  simulated_y_pred[:, :-token_win-num_token_heads+1],
                                                                  reduced_size,
                                                                  [max(y_len) for _ in range(b)],
                                                                  features_size,
                                                                  start=0,
                                                                  cache=cache,
                                                                  keep_all_weights=True)

            pred = pred.permute(0, 2, 1, 3).reshape(pred.size(0), pred.size(1)*pred.size(2), pred.size(3)).permute(0, 2, 1)
            gt = torch.cat([y[:, token_win-1+i:y.size(1)-(num_token_heads-i)] for i in range(1, num_token_heads+1)], dim=1)
            loss_ce = loss_func(pred, gt)
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
        num_token_heads = self.models["decoder"].num_token_heads
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        max_chars = self.params["training_params"]["max_char_prediction"]

        decoding_mt_mode = self.params["model_params"]["decoding_mt_mode"]
        decoding_num_preds = self.params["model_params"]["decoding_num_preds"]
        decoding_threshold = self.params["model_params"]["decoding_threshold"]
        decoding_win_mode = self.params["model_params"]["decoding_win_mode"]

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, token_win), dtype=torch.long, device=self.device) * self.tokenizer.tokens["<start>"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device) * token_win

            all_weights = list()
            all_preds = list()
            confidence_scores = list()
            cache = None

            features = self.models["encoder"](x)
            features_size = features.size()
            pos_features = self.models["decoder"].get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            num_new_preds = token_win
            while predicted_tokens.size(1) < max_chars:
                output, pred, cache, weights = self.models["decoder"](features,
                                                                      predicted_tokens,
                                                                      reduced_size,
                                                                      predicted_tokens_len,
                                                                      features_size,
                                                                      start=0,
                                                                      cache=cache,
                                                                      num_pred=num_new_preds,
                                                                      )

                if return_weights:
                    all_preds.append(torch.argmax(pred[:, -token_win:, :], dim=-1, keepdim=False).detach().cpu())
                    all_weights.append(weights.detach().cpu())
                confidence_scores.append(torch.max(torch.softmax(pred, dim=-1), dim=-1).values.cpu().detach().numpy())
                new_preds = pred[:, :, -num_token_heads - token_win + 1:]

                # Naive approach: keep prediction from first head for all tokens expect the last one for which we apply decoding_mt_mode
                if decoding_win_mode == "first":
                    new_preds_list = [torch.argmax(new_preds[:, -token_win:-1, 0], dim=-1), ]
                    if decoding_mt_mode == "static":
                        new_preds_list.append(torch.argmax(new_preds[:, -1, 0:decoding_num_preds], dim=-1))
                    else:  # decoding_mt_mode == "dynamic" (designed for batch size == 1 only)
                        new_preds_list.append(torch.argmax(pred[:, -1, -num_token_heads], dim=-1).unsqueeze(-1))
                        for i in range(1, num_token_heads):
                            if torch.max(pred[:, -1, -num_token_heads + i], dim=-1).values > decoding_threshold:
                                new_preds_list.append(torch.argmax(pred[:, -1, -num_token_heads + i], dim=-1).unsqueeze(-1))
                            else:
                                break
                else:
                    raise NotImplementedError()

                new_preds = torch.cat(new_preds_list, dim=-1)
                num_new_preds = new_preds.size(-1)
                predicted_tokens = torch.cat([predicted_tokens, new_preds], dim=1)
                new_ending = torch.sum(torch.eq(new_preds, self.tokenizer.tokens["<end>"]), dtype=bool, dim=-1, keepdim=True)
                reached_end = torch.logical_or(reached_end, new_ending)
                predicted_tokens_len += num_new_preds

                if torch.all(reached_end):
                    break

            predicted_tokens = predicted_tokens[:, token_win:]
            for i in range(b):
                end_token_indices = torch.nonzero(predicted_tokens[i] == self.tokenizer.tokens["<end>"])
                prediction_len[i] = max_chars - 1 if end_token_indices.numel() == 0 else end_token_indices[0]
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]].cpu().numpy() for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]
        if return_weights:
            return str_x, confidence_scores, all_weights, all_preds
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
            str_x, confidence, attention_weights, all_preds = self.inference_batch(batch_data, return_weights=True)
            win_mode = self.params["model_params"]["decoding_win_mode"]
            mt_mode = self.params["model_params"]["decoding_mt_mode"]
            threshold = self.params["model_params"]["decoding_threshold"]
            num_preds = self.params["model_params"]["decoding_num_preds"]
            token_win = self.params["model_params"]["token_win"] # W
            for i in range(len(all_preds)):
                pred_at_step_i = all_preds[i]  # (B, W, N)
                all_preds[i] = []
                for k in range(token_win):
                    all_preds[i].append([self.tokenizer.token_to_text([t, ]) if t != self.tokenizer.tokens["<end>"] else "<end>" for t in pred_at_step_i[0][k]])
            img = image_torch_to_numpy(batch_data["imgs"][0])
            attention_weights = [a[0, -token_win:].numpy() for a in attention_weights]
            attention_weights = format_attention_weights(attention_weights, img.shape[:2])
            confidence = [conf[0] for conf in confidence]
            generate_meta_dan_visualization(img, attention_weights, all_preds, confidence, output_attention_path, win_mode, mt_mode, threshold, num_preds)
        else:
            str_x, confidence = self.inference_batch(batch_data, return_weights=False)

        return str_x
    
    def eval_all(self, set_name="test"):
        """
        Evaluate on set_name with many decoding modes
        """
        # thresholds = [a/1000 for a in list(range(0, 1001, 50))]
        thresholds = [0.9, ]
        paths = []

        for dataset in self.params["dataset_params"][set_name]:
            for strategy in ["first", "average", "majority"]:
                self.params["model_params"]["decoding_win_mode"] = strategy
                self.params["model_params"]["decoding_mt_mode"] = "static"
                self.params["model_params"]["decoding_threshold"] = 0.9

                list_num_preds = range(1, self.params["model_params"]["num_token_heads"]+1) if strategy == "first" else [self.params["model_params"]["num_token_heads"], ]
                for num_preds in list_num_preds:
                    self.params["model_params"]["decoding_num_preds"] = num_preds
                    self.output_file_suffix = f"_{strategy}_{self.params['model_params']['decoding_mt_mode']}_{num_preds}"
                    path = self.get_output_file_path(dataset['name'])
                    if not os.path.exists(path):
                        self.evaluate(dataset["name"], split_name=set_name, output=True)

                self.params["model_params"]["decoding_mt_mode"] = "dynamic"
                for threshold in thresholds:
                    self.params["model_params"]["decoding_threshold"] = threshold
                    self.output_file_suffix = f"_{strategy}_{self.params['model_params']['decoding_mt_mode']}_{threshold}"
                    paths.append(self.get_output_file_path(dataset['name']))
                    if not os.path.exists(paths[-1]):
                        self.evaluate(dataset["name"], split_name=set_name, output=True)


        from dar.text_recognition.document.mt_dan.result_viz import generate_graph_result
        generate_graph_result(paths, "cer", thresholds)