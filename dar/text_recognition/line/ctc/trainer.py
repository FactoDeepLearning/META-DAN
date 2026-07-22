from dar.text_recognition.ocr_manager import OCRManager
import torch
from torch import autocast
from torch.nn import CTCLoss
import re
import time


class LineCTCManager(OCRManager):

    def __init__(self, params, load_dataset=True):
        super().__init__(params, load_dataset)

    def train_batch(self, batch_data, metric_names):
        """
        Forward and backward pass for training
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.tokenizer.tokens["<blank>"])

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)
        self.step_optimizers()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": self.pred_to_str(pred, x_reduced_len),
            "str_y": batch_data["str_labels"]
        }

        values = self.add_syn_values(values)

        return values

    def inference_batch(self, batch_data):
        x = batch_data["imgs"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        with autocast(device_type=self.autocast_mode, enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        str_x = self.pred_to_str(pred, x_reduced_len)
        return str_x

    def evaluate_batch(self, batch_data, metric_names):
        """
        Forward pass only for validation and test
        """
        start_time = time.time()
        str_x = self.inference_batch(batch_data)
        process_time = time.time() - start_time

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "str_x": str_x,
            "str_y": batch_data["str_labels"],
            "time": process_time
        }
        return values

    def predict_batch(self, batch_data):
        return self.inference_batch(batch_data)

    def pred_to_str(self, pred, pred_len):
        """
        convert prediction tokens to string
        """
        ind_x = [pred[i][:pred_len[i]] for i in range(pred.shape[0])]
        str_x = [self.tokenizer.token_to_text(self.tokenizer.preprocess_ctc(t)) for t in ind_x]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        return str_x