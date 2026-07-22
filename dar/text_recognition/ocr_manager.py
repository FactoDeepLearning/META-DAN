from dar.generic.training_manager import GenericTrainingManager
from dar.generic.dataset_manager import GenericDataset
from dar.text_recognition.ocr_dataset_manager import OCRDataset, OCRCollateFunction
import numpy as np
import os
import torch
from tqdm import tqdm


class OCRManager(GenericTrainingManager):
    def __init__(self, params, load_dataset=True):
        super(OCRManager, self).__init__(params, load_dataset)
        if "vocab_size" not in self.params["model_params"]:
            self.params["model_params"]["vocab_size"] = self.tokenizer.num_tokens

    def add_syn_values(self, values: dict) -> dict:
        """
        Add metrics related to synthetic document generation process for logs.
        """
        if "syn_max_lines" in self.params["training_params"]["train_metrics"]:
            syn_config = self.dataset.train_dataset.params["config"]["synthetic_data"]
            syn_generator_configs_per_dataset = syn_config["custom_config"] if "custom_config" in syn_config else None
            if syn_generator_configs_per_dataset is not None:
                max_lines_per_dataset = []
                for dataset_config in syn_generator_configs_per_dataset.values():
                    max_lines_per_dataset.append(np.max([self.dataset.train_dataset.synthetic_manager.get_syn_max_lines(absolute_max_lines=config["max_num_lines"]) for config in dataset_config]))         
                values["syn_max_lines"] = np.max(max_lines_per_dataset)
            else:
                values["syn_max_lines"] = self.dataset.train_dataset.synthetic_manager.get_syn_max_lines()
        if "syn_ratio" in self.params["training_params"]["train_metrics"]:
            values["syn_ratio"] = self.dataset.train_dataset.synthetic_manager.get_syn_ratio()
        return values

    def predict(self, source: str, target: str, output_attention: bool=False) -> None:
        """
        Method to perform HTR prediction with current loaded model.
        :param source: Path of the image on which to perform the recognition (or a folder path to process all images in of that folder).
        :param target: Path of the folder in which to store the results of the recognition.
        :param output_attention: Whether to produce a video showing the attention maps of the different decoding steps.
        """
        os.makedirs(target, exist_ok=True)
        if os.path.isfile(source):
            source = [source]
        else:
            source = tqdm([os.path.join(source, filename) for filename in os.listdir(source)])

        dparams = self.params["dataset_params"]
        reduce_dims = np.array([dparams["config"]["height_divisor"], dparams["config"]["width_divisor"], 1])
        fn = OCRCollateFunction(dparams["config"])

        for img_path in source:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            run_name = os.path.basename(self.paths["output_folder"])
            out_fold = os.path.join(target, run_name)
            os.makedirs(out_fold, exist_ok=True)
            output_path = os.path.join(out_fold, f"{stem}.txt")
            img = GenericDataset.load_image(img_path)

            sample = {
                "img": img,
                "name": "notused",
                "token_label": torch.zeros((20, )),
                "label_len": 20,
            }
            sample = OCRDataset.prepare_sample(sample, set_name="test", params=dparams, reduce_dims_factor=reduce_dims, padding_value=dparams["config"]["image_padding_value"], include_preprocessing=True)
            batch_sample = fn([sample])
            for model_name in self.models.keys():
                self.models[model_name].eval()
            with torch.no_grad():
                if output_attention:
                    attention_path = os.path.join(out_fold, f"{stem}.mp4")
                    str_x = self.predict_batch(batch_sample, output_attention_path=attention_path)[0]
                else:
                    str_x = self.predict_batch(batch_sample)[0]
                with open(output_path, "w") as fout:
                    fout.write(str_x)