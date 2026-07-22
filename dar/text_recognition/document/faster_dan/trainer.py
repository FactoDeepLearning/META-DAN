from dar.text_recognition.ocr_manager import OCRManager
from dar.text_recognition.document.faster_dan.attention_visualization import generate_faster_dan_attention
from dar.generic.utils_visualization import format_attention_weights
from dar.generic.utils import image_torch_to_numpy
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from torch import autocast
import time


class FasterDANManager(OCRManager):

    def __init__(self, params, load_dataset=True):
        super().__init__(params, load_dataset)

    def split_tokens_two_step(self, tokens):
        linebreak_index = self.tokenizer.tokens["\n"] if "\n" in self.tokenizer.tokens else -1
        end_line_token = torch.tensor(self.tokenizer.tokens["<end>"], device=tokens.device, dtype=tokens.dtype)
        batch_token_list = list()
        for k in range(tokens.size(0)):
            first_pass_tokens = list()
            lines_tokens = list()
            line_tokens = list()
            for i in range(tokens.size(1)):
                if tokens[k, i] == self.tokenizer.tokens["<pad>"]:
                    continue
                if tokens[k, i] >= self.tokenizer.num_text_tokens or (i > 0 and (tokens[k, i-1] >= self.tokenizer.num_text_tokens or tokens[k, i-1] == linebreak_index)):
                    first_pass_tokens.append(tokens[k, i])
                    if len(line_tokens) > 0:
                        line_tokens.append(end_line_token)
                        lines_tokens.append(line_tokens)
                        line_tokens = list()
                elif tokens[k, i] == linebreak_index:
                    line_tokens.append(end_line_token)
                    lines_tokens.append(line_tokens)
                    line_tokens = list()
                    continue
                if tokens[k, i] < self.tokenizer.num_text_tokens:
                    line_tokens.append(tokens[k, i])
            if len(line_tokens) > 0:
                lines_tokens.append(line_tokens)
            first_pass = torch.stack(first_pass_tokens)
            lines_tokens = [torch.stack(l) for l in lines_tokens]
            lines_tokens.insert(0, first_pass)

            tf_tokens = torch.cat([l[:-1] for l in lines_tokens])
            gt_tokens = torch.cat([l[1:] for l in lines_tokens])
            line_indices = torch.cat([(torch.ones(lines_tokens[i].size(), device=tokens.device, dtype=tokens.dtype)*i)[:-1] for i in range(len(lines_tokens))])
            index_in_lines = torch.cat([torch.arange(0, lines_tokens[i].size(0), device=tokens.device, dtype=tokens.dtype)[:-1] for i in range(len(lines_tokens))])
            batch_token_list.append((tf_tokens, gt_tokens, line_indices, index_in_lines))
        max_len = max([b[0].size(0) for b in batch_token_list])
        padded_batch_tokens = list()
        for i in range(len(batch_token_list[0])):
            padded_tokens = torch.ones((tokens.size(0), max_len), device=tokens.device, dtype=tokens.dtype) * self.tokenizer.tokens["<pad>"]
            for k in range(tokens.size(0)):
                padded_tokens[k, :batch_token_list[k][i].size(0)] = batch_token_list[k][i]
            padded_batch_tokens.append(padded_tokens)
        return padded_batch_tokens

    def merge_tokens_two_step(self, tokens, line_indices, y):
        new_tokens = list()
        for b in range(len(tokens)):
            lines_tokens = list()
            num_first_pass = int(np.sum(line_indices[b] == 0))
            indices = np.argwhere(y[b, :num_first_pass] < self.tokenizer.num_text_tokens)[:, 0]
            for i in range(tokens[b].shape[0]):
                line_index = int(line_indices[b, i])
                if line_index == 0:
                    lines_tokens.append([tokens[b][i], ])
                else:
                    lines_tokens[indices[line_index-1]].append(tokens[b][i])

            for i in range(len(lines_tokens)-1):
                if len(lines_tokens[i]) > 1 and len(lines_tokens[i+1]) > 1:
                    lines_tokens[i].append(self.tokenizer.tokens["\n"])
            new_tokens.append(np.concatenate(lines_tokens, axis=0))
        return new_tokens

    def is_end_layout_token(self, token):
        end_layout_tokens = [self.tokenizer.tokens[t] for t in self.metric_manager["train"].matching_tokens.values()]
        return token in [self.tokenizer.tokens["\n"], self.tokenizer.tokens["<end>"]] or token in end_layout_tokens

    def add_error_in_gt(self, y, y_len, error_rate, preserve_layout=False):
        linebreak_index = self.tokenizer.tokens["\n"] if "\n" in self.tokenizer.tokens else -1
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate:
                    if preserve_layout:
                        if y[b][i] != linebreak_index and y[b][i] < self.tokenizer.num_text_tokens:
                            new_token = linebreak_index
                            while new_token == linebreak_index:
                                new_token = np.random.randint(0, self.tokenizer.num_text_tokens)
                        else:
                            new_token = y_error[b][i]
                    else:
                        new_token = np.random.randint(0, self.tokenizer.num_tokens)
                    y_error[b][i] = new_token
        return y_error

    def train_batch(self, batch_data, metric_names):
        loss_reduction = self.params["training_params"]["loss_reduction"] if "loss_reduction" in self.params["training_params"] else "sum"
        loss_ce = CrossEntropyLoss(ignore_index=self.tokenizer.tokens["<pad>"], reduction=loss_reduction)

        x = batch_data["imgs"].to(self.device)
        y = batch_data["fdan_labels"].to(self.device)
        line_indices = batch_data["fdan_line_indices"][:, :-1].to(self.device)
        index_in_lines = batch_data["fdan_index_in_lines"][:, :-1].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate, preserve_layout=True)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate, preserve_layout=True)
        else:
            simulated_y_pred = y

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            raw_features = self.models["encoder"](x)
            features_size = raw_features.size()
            b, c, h, w = features_size

            features = self.models["decoder"].get_pos_features(raw_features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)
            y = y[:, 1:]
            simulated_y_pred = simulated_y_pred[:, :-1]
            y[simulated_y_pred == self.tokenizer.tokens["<end>"]] = self.tokenizer.tokens["<pad>"]
            token_len = [int(torch.where(y_i == self.tokenizer.tokens["<pad>"])[0][0]) if torch.where(y_i == self.tokenizer.tokens["<pad>"])[0].size(0) > 0 else y_i.size(0) for y_i in simulated_y_pred]
            output, pred, cache, weights = self.models["decoder"](features,
                                                                  simulated_y_pred,
                                                                  reduced_size,
                                                                  token_len,
                                                                  features_size,
                                                                  keep_all_weights=True,
                                                                  line_indices=line_indices,
                                                                  index_in_lines=index_in_lines,
                                                                  padding_value=self.tokenizer.tokens["<pad>"])

            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :token_len[i]] for i in range(b)]
            predicted_tokens = self.merge_tokens_two_step(predicted_tokens, line_indices.detach().cpu().numpy(), y.detach().cpu().numpy())

            loss = loss_ce(pred, y)

            with autocast(device_type=self.autocast_mode, enabled=False):
                self.backward_loss(loss)
                self.step_optimizers()

            str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": batch_data["str_labels"],
            "str_x": str_x,
            "loss": loss.item(),
        }

        values = self.add_syn_values(values)

        return values


    def eval_first_pass(self, features, reduced_size, features_size, return_weights=False):
        max_line_pred = self.params["training_params"]["max_line_pred"]
        b = features.size(1)
        cache = None
        whole_output = list()
        global_weights = list()
        confidence_scores = torch.zeros((b, 1), dtype=torch.float, device=self.device)
        line_indices = torch.zeros((b, 1), dtype=torch.long, device=self.device)
        index_in_lines = torch.zeros((b, 1), dtype=torch.long, device=self.device)
        predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.tokenizer.tokens["<start>"]
        prediction_len = torch.ones((b,), dtype=torch.int, device=self.device)
        reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
        for i in range(1, max_line_pred+1):
            output, pred, cache, weights = self.models["decoder"](features,
                                                                  predicted_tokens,
                                                                  reduced_size,
                                                                  prediction_len,
                                                                  features_size,
                                                                  cache=cache,
                                                                  num_pred=1,
                                                                  line_indices=line_indices,
                                                                  index_in_lines=index_in_lines,
                                                                  padding_value=self.tokenizer.tokens["<pad>"])
            whole_output.append(output)
            if return_weights:
                global_weights.append(weights)
            confidence_scores = torch.cat([confidence_scores, torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values], dim=1)
            new_predicted_tokens = torch.argmax(pred[:, :, -1], dim=1, keepdim=True)
            new_predicted_tokens[reached_end] = self.tokenizer.tokens["<pad>"]
            predicted_tokens = torch.cat([predicted_tokens, new_predicted_tokens], dim=1)

            line_indices = torch.zeros(predicted_tokens.size(), device=line_indices.device, dtype=line_indices.dtype)
            index_in_lines = torch.stack([torch.arange(0, predicted_tokens.size(1), device=index_in_lines.device, dtype=index_in_lines.dtype) for _ in range(predicted_tokens.size(0))])

            prediction_len[reached_end == False] = i + 1
            reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.tokenizer.tokens["<end>"]))
            if torch.all(reached_end):
                break
        prediction_len[torch.eq(reached_end, False)] = max_line_pred
        predicted_tokens = predicted_tokens[:, 1:-1]
        confidence_scores = confidence_scores[:, 1:-1]
        cache = cache[:, :-1]
        prediction_len -= 2
        global_weights = torch.cat(global_weights[:-1], dim=1) if return_weights else None
        return predicted_tokens, prediction_len, confidence_scores, cache, global_weights

    def eval_second_pass(self, features, reduced_size, features_size, predicted_tokens, prediction_len, confidence_scores, cache, first_pass_weights=None):
        linebreak_index = self.tokenizer.tokens["\n"] if "\n" in self.tokenizer.tokens else -1
        max_pred_per_line = self.params["training_params"]["max_pred_per_line"]
        b = features.size(1)
        global_weights = torch.cat([first_pass_weights, first_pass_weights], dim=1) if first_pass_weights is not None else None
        line_indices = torch.zeros(predicted_tokens.size(), device=predicted_tokens.device, dtype=predicted_tokens.dtype)
        index_in_lines = torch.stack([torch.arange(0, predicted_tokens.size(1), device=predicted_tokens.device, dtype=predicted_tokens.dtype) for _ in range(b)], dim=0)
        line_indices2 = torch.where(predicted_tokens[:, :] < self.tokenizer.num_text_tokens, 1, 0)
        line_indices2 = torch.cumsum(line_indices2, dim=1)
        line_indices = torch.cat([line_indices, line_indices2], dim=1)
        index_in_lines = torch.cat([index_in_lines, torch.zeros(index_in_lines.size(), device=index_in_lines.device,
                                                                dtype=index_in_lines.dtype)], dim=1)
        num_lines = predicted_tokens.size(1)
        if num_lines == 0:
            return np.empty((b, 0)), np.empty((b, 0))
        predicted_tokens = torch.cat([predicted_tokens, predicted_tokens], dim=1)
        index_to_pad = torch.logical_and(predicted_tokens >= self.tokenizer.num_text_tokens, line_indices!=0)
        predicted_tokens[index_to_pad] = self.tokenizer.tokens["<pad>"]
        num_lines_per_sample = prediction_len
        prediction_len = prediction_len * 2
        confidence_scores = torch.cat([confidence_scores, confidence_scores], dim=1)
        prediction_len_line = torch.zeros((b, num_lines), dtype=torch.int, device=self.device)
        reached_end_line = torch.zeros((b, num_lines), dtype=torch.bool, device=self.device)
        for bi in range(b):
            for k in range(num_lines):
                if predicted_tokens[bi, k] >= self.tokenizer.num_text_tokens:
                    reached_end_line[bi, k] = True
                    prediction_len_line[bi, k] = 1

        for i in range(0, max_pred_per_line):
            model = self.models["decoder"]
            num_pred = num_lines

            output, pred, cache, weights = model(features,
                                                 predicted_tokens,
                                                 reduced_size,
                                                 prediction_len,
                                                 features_size,
                                                 cache=cache,
                                                 num_pred=num_pred,
                                                 line_indices=line_indices,
                                                 index_in_lines=index_in_lines,
                                                 two_step_parallel=True,
                                                 padding_value=self.tokenizer.tokens["<pad>"]
                                                 )
            if first_pass_weights is not None:
                global_weights = torch.cat([global_weights, weights], dim=1)
            confidence_scores = torch.cat([confidence_scores, torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values], dim=1)
            new_predicted_tokens = torch.argmax(pred[:, :, -num_lines:], dim=1, keepdim=False)
            new_predicted_tokens[reached_end_line] = self.tokenizer.tokens["<pad>"]
            predicted_tokens = torch.cat([predicted_tokens, new_predicted_tokens], dim=1)
            index_in_lines = torch.cat([index_in_lines, index_in_lines[:, -num_lines:] + 1], dim=1)
            line_indices = torch.cat([line_indices, line_indices[:, num_lines:2 * num_lines]], dim=1)
            prediction_len += num_lines
            prediction_len_line[reached_end_line == False] = i + 1
            reached_end_line = torch.logical_or(reached_end_line, torch.eq(predicted_tokens[:, -num_lines:], self.tokenizer.tokens["<end>"]))
            if torch.all(reached_end_line):
                break

        line_break_token = torch.ones((1,), device=predicted_tokens.device, dtype=predicted_tokens.dtype) * linebreak_index
        line_break_confidence = torch.ones((1,), device=confidence_scores.device, dtype=confidence_scores.dtype)
        prediction_len_line[torch.eq(reached_end_line, False)] = max_pred_per_line + 1
        batch_pred = list()
        for k in range(b):
            list_line_tokens = list()
            list_line_scores = list()
            list_line_weights = list()
            for l in range(num_lines_per_sample[k]):
                line_tokens = torch.cat([predicted_tokens[k, l:l+1], predicted_tokens[k, 2*num_lines + l::num_lines]], dim=0)
                line_tokens = line_tokens[:prediction_len_line[k, l]]
                list_line_tokens.append(line_tokens)
                line_scores = torch.cat([confidence_scores[k, l:l+1], confidence_scores[k, 2*num_lines + l::num_lines]], dim=0)
                line_scores = line_scores[:prediction_len_line[k, l]]
                list_line_scores.append(line_scores)
                if first_pass_weights is not None:
                    line_weights = global_weights[k, num_lines + l::num_lines]
                    line_weights = line_weights[:prediction_len_line[k, l]]
                    list_line_weights.append(line_weights)
            batch_pred.append([list_line_tokens, list_line_scores, list_line_weights])
        if first_pass_weights is None:
            tokens = list()
            scores = list()
            for k in range(b):
                list_line_tokens, list_line_scores, list_line_weights = batch_pred[k]
                for l in range(len(list_line_tokens) - 1):
                    if list_line_tokens[l][0] < self.tokenizer.num_text_tokens and list_line_tokens[l + 1][0] < self.tokenizer.num_text_tokens:
                        list_line_tokens[l] = torch.cat([list_line_tokens[l], line_break_token], dim=0)
                        list_line_scores[l] = torch.cat([list_line_scores[l], line_break_confidence], dim=0)

                list_line_tokens = torch.cat(list_line_tokens, dim=0).cpu().numpy() if len(list_line_tokens) > 0 else list()
                tokens.append(list_line_tokens)
                list_line_scores = torch.cat(list_line_scores, dim=0) if len(list_line_scores) > 0 else list()
                if "<blank>" in self.tokenizer.tokens:
                    list_line_scores = list_line_scores[list_line_tokens != self.tokenizer.tokens["<blank>"]]
                scores.append(list_line_scores.cpu().detach().numpy())
            return scores, tokens
        line_pred = [a[0] for a in batch_pred]
        line_weights = [a[2] for a in batch_pred]
        return line_pred, line_weights

    def inference(self, batch_data, return_weights=False):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            features = self.models["encoder"](x)
            features_size = features.size()
            features = self.models["decoder"].get_pos_features(features)
            features = torch.flatten(features, start_dim=2, end_dim=3).permute(2, 0, 1)

            predicted_tokens, prediction_len, confidence_scores, cache, first_pass_weights = self.eval_first_pass(features, reduced_size, features_size, return_weights=return_weights)

            _, y, y_line_ind, _ = self.split_tokens_two_step(batch_data["labels"])

            return self.eval_second_pass(features, reduced_size, features_size,
                                                             predicted_tokens, prediction_len,
                                                             confidence_scores, cache,
                                                             first_pass_weights=first_pass_weights)

    def evaluate_batch(self, batch_data, metric_names):
        start_time = time.time()
        scores, predicted_tokens = self.inference(batch_data)
        str_x = [self.tokenizer.token_to_text(t) for t in predicted_tokens]

        process_time = time.time() - start_time
        values = {
            "nb_samples": len(batch_data["imgs"]),
            "str_y": batch_data["str_labels"],
            "str_x": str_x,
            "confidence_score": scores,
            "time": process_time,
        }

        return values

    def predict_batch(self, batch_data, output_attention_path=None):
        line_pred, line_weights = self.inference(batch_data, return_weights=True)

        line_str_pred = [self.tokenizer.token_to_text(line) for line in line_pred[0]]

        if output_attention_path:
            img = image_torch_to_numpy(batch_data["imgs"][0])
            line_weights = [l.detach().cpu().numpy() for l in line_weights[0]]
            line_weights = [format_attention_weights([l], img.shape[:2])[0] for l in line_weights]
            generate_faster_dan_attention(img, line_weights, line_str_pred, output_attention_path)
        return ["\n".join(line_str_pred)]
