from dar.text_recognition.ocr_manager import OCRManager
from dar.text_recognition.document.mt_dan.attention_visualization import generate_mt_dan_visualization
from dar.generic.utils_visualization import format_attention_weights
from dar.generic.utils import image_torch_to_numpy
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from torch import autocast
import time
import os


class MTDANManager(OCRManager):
    def __init__(self, params, load_dataset=True):
        super().__init__(params, load_dataset)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.tokenizer.tokens["<pad>"]:
                    y_error[b][i] = np.random.randint(0, self.tokenizer.num_tokens)
        return y_error, y_len

    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.tokenizer.tokens["<pad>"])
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

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
            num_token_heads = self.models["decoder"].num_token_heads

            features = self.models["decoder"].get_pos_features(raw_features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, cache, weights = self.models["decoder"](features,
                                                                  simulated_y_pred[:, :-num_token_heads],
                                                                  reduced_size,
                                                                  y_len,
                                                                  features_size,
                                                                  start=0,
                                                                  cache=cache,
                                                                  keep_all_weights=True)

            pred = pred.permute(0, 2, 1, 3).reshape(pred.size(0), pred.size(1)*pred.size(2), pred.size(3)).permute(0, 2, 1)  # (B, L, m, C) -> (B, C, L*m)
            gt = torch.cat([y[:, i:y.size(1)-(num_token_heads-i)] for i in range(1, num_token_heads+1)], dim=1) # [y, y with shift 1, y with shift 2, etc...]
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
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]
        num_token_heads = self.models["decoder"].num_token_heads

        decoding_mt_mode = self.params["model_params"]["decoding_mt_mode"]
        decoding_num_preds = self.params["model_params"]["decoding_num_preds"]
        decoding_threshold = self.params["model_params"]["decoding_threshold"]

        if decoding_mt_mode == "static":
            assert 0 < decoding_num_preds <= num_token_heads

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b,), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.tokenizer.tokens["<start>"]
            predicted_tokens_len = torch.ones((b,), dtype=torch.int, device=self.device)

            all_weights = list()
            all_preds = list()
            confidence_scores = list()
            cache = None

            features = self.models["encoder"](x)
            features_size = features.size()
            features = self.models["decoder"].get_pos_features(features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)

            num_new_preds = 1
            while predicted_tokens.size(1) < max_chars:
                output, pred, cache, weights = self.models["decoder"](features,
                                                                      predicted_tokens,
                                                                      reduced_size,
                                                                      predicted_tokens_len,
                                                                      features_size,
                                                                      start=0,
                                                                      cache=cache,
                                                                      num_pred=num_new_preds)
                pred = torch.softmax(pred, dim=-1)
                confidence_scores.append(torch.max(pred, dim=-1).values)
                if return_weights:
                    all_preds.append(torch.argmax(pred[:, -1, :], dim=-1, keepdim=False).detach().cpu())
                    all_weights.append(weights.detach().cpu())
                if decoding_mt_mode == "static":
                    new_preds = torch.argmax(pred[:, -1, 0:decoding_num_preds], dim=-1, keepdim=False)
                else:  # decoding_mt_mode == "dynamic" (designed for batch size == 1 only)
                    new_preds = [torch.argmax(pred[:, -1, -num_token_heads], dim=-1)]
                    for i in range(1, num_token_heads):
                        if torch.max(pred[:, -1, -num_token_heads + i], dim=-1).values > decoding_threshold:
                            new_preds.append(torch.argmax(pred[:, -1, -num_token_heads + i], dim=-1))
                        else:
                            break
                    new_preds = torch.stack(new_preds, dim=1)

                num_new_preds = new_preds.size(1)
                predicted_tokens = torch.cat([predicted_tokens, new_preds], dim=1)
                new_end, index_end_tokens = torch.max(torch.eq(new_preds, self.tokenizer.tokens["<end>"]), dim=-1)
                mask_first_end = torch.logical_and(torch.logical_not(reached_end), new_end)
                reached_end = torch.logical_or(reached_end, new_end)
                prediction_len[mask_first_end] = predicted_tokens_len[mask_first_end] + index_end_tokens[mask_first_end].to(torch.int)
                predicted_tokens_len += num_token_heads

                if torch.all(reached_end):
                    break

            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            prediction_len[torch.eq(reached_end, False)] = predicted_tokens.size(1) - 1
            predicted_tokens_list = [predicted_tokens[i, 1:prediction_len[i]].cpu().numpy() for i in range(b)]
            confidence_scores_list = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens_list]
        if return_weights:
            return str_x, confidence_scores_list, all_weights, all_preds
        return str_x, confidence_scores_list

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
        decoding_mt_mode = self.params["model_params"]["decoding_mt_mode"]
        decoding_num_preds = self.params["model_params"]["decoding_num_preds"]
        decoding_threshold = self.params["model_params"]["decoding_threshold"]
        str_x, conf_scores, attention_weights, all_preds = self.inference_batch(batch_data, return_weights=True)

        if output_attention_path:
            for i in range(len(all_preds)):
                all_preds[i] = [self.tokenizer.token_to_text([t, ]) if t != self.tokenizer.tokens["<end>"] else "<end>" for t in all_preds[i][0]]
            img = image_torch_to_numpy(batch_data["imgs"][0])
            attention_weights = format_attention_weights([a[0].numpy() for a in attention_weights], img.shape[:2])
            attention_weights = np.concatenate(attention_weights, axis=0)
            generate_mt_dan_visualization(img, attention_weights, all_preds, output_attention_path, conf_scores[0], decoding_mt_mode, decoding_num_preds, decoding_threshold)

        return str_x

    def eval_all(self, set_name="test"):
        """
        Evaluate on set_name with all decoding modes
        """
        thresholds = [a/1000 for a in list(range(0, 1001, 50))]
        paths = []

        for dataset in self.params["dataset_params"][set_name]:
            self.params["model_params"]["decoding_mt_mode"] = "static"
            for num_preds in range(1, self.params["model_params"]["num_token_heads"]+1):
                self.params["model_params"]["decoding_num_preds"] = num_preds
                self.output_file_suffix = f"_{self.params['model_params']['decoding_mt_mode']}_{num_preds}"
                path = self.get_output_file_path(dataset['name'])
                if not os.path.exists(path):
                    self.evaluate(dataset["name"], split_name=set_name, output=True)

            self.params["model_params"]["decoding_mt_mode"] = "dynamic"
            for threshold in thresholds:
                self.params["model_params"]["decoding_threshold"] = threshold
                self.output_file_suffix = f"_{self.params['model_params']['decoding_mt_mode']}_{threshold}"
                paths.append(self.get_output_file_path(dataset['name']))
                if not os.path.exists(paths[-1]):
                    self.evaluate(dataset["name"], split_name=set_name, output=True)


        from dar.text_recognition.document.mt_dan.result_viz import generate_graph_result
        generate_graph_result(paths, "cer", thresholds)