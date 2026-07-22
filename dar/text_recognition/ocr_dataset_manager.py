from dar.generic.dataset_manager import DatasetManager, GenericDataset
from dar.generic.utils import pad_images, pad_image_width_right, pad_sequences_1D, pad_image, pad_image_height_bottom
from dar.generic.utils import randint, list_split
from dar.generic.dataset_manager import apply_preprocessing
from dar.text_recognition.synthetic.synthetic_manager import SyntheticManager

from torchvision.transforms import ToTensor
import os
import copy
import pickle
import numpy as np
import torch
from PIL import Image
from bidi.algorithm import get_display


class OCRDatasetManager(DatasetManager):
    """
    Specific class to handle text_recognition/HTR tasks
    """

    def __init__(self, params):
        super(OCRDatasetManager, self).__init__(params)

        charset = params["charset"] if "charset" in params else self.get_merged_charsets()

        additional_tokens = list()
        if self.params["config"]["charset_mode"].lower() == "ctc":
            additional_tokens.append("<blank>")
        elif self.params["config"]["charset_mode"] == "seq2seq":
            additional_tokens.append("<end>")
            additional_tokens.append("<start>")

        if "add_eot" in self.params["config"]["constraints"] and "<end>" not in additional_tokens:
            additional_tokens.append("<end>")
        if "add_sot" in self.params["config"]["constraints"] and "<start>" not in additional_tokens:
            additional_tokens.append("<start>")
        additional_tokens.append("<pad>")

        self.tokenizer = self.params["tokenizer_class"]({
            "charset": charset,
            "additional_tokens": additional_tokens,
            "dataset_paths": [os.path.join(self.params["dataset_folder"], dataset["dataset_name"]) for dataset in self.params["train"]],
            "bpe_source": self.params["tokenizer_source"],
            "tok_num_tokens": self.params["tokenizer_num_tokens"],
            "dataset_name": "-".join([d["dataset_name"] for d in params["train"]]),
            "model_name": self.params["tokenizer_model_name"],
        })

        self.params["config"]["padding_token"] = self.tokenizer.tokens["<pad>"]

    def get_merged_charsets(self):
        """
        Merge the charset of the different datasets used
        """
        dataset_names = [set["dataset_name"] for set in self.params["train"]]
        charset = set()
        for dataset_name in dataset_names:
            dataset_charset_path = os.path.join(self.params["dataset_folder"], dataset_name, "charset.pkl")
            with open(dataset_charset_path, "rb") as f:
                current_charset = pickle.load(f)
                charset = charset.union(set(current_charset))
        if "\n" in charset and "remove_linebreaks" in self.params["config"]["constraints"]:
            charset.remove("\n")
        if "" in charset:
            charset.remove("")
        return sorted(list(charset))

    def apply_specific_treatment_after_dataset_loading(self, dataset: GenericDataset) -> None:
        """
        Update dataset features that requires all sets to be loaded first
        """
        dataset.tokenizer = self.tokenizer
        dataset.convert_labels()
        if "READ_2016" in dataset.name and "augmentation" in dataset.params["config"] and dataset.params["config"]["augmentation"]:
            dataset.params["config"]["augmentation"]["fill_value"] = tuple([int(i) for i in dataset.mean])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_height"] == "max":
            dataset.params["config"]["padding"]["min_height"] = max([s["img"].shape[0] for s in self.train_dataset.samples])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_width"] == "max":
            dataset.params["config"]["padding"]["min_width"] = max([s["img"].shape[1] for s in self.train_dataset.samples])

        if dataset.use_syn:
            dataset.synthetic_manager = SyntheticManager(dataset)
        if dataset.make_syn_val:
            for i in range(len(dataset.samples)):
                dataset.samples[i] = dataset.generate_synthetic_sample(dataset.samples[i])
                if not dataset.params["config"]["load_in_memory"]:
                    img_path = os.path.join(".cache", "dataset-syn-val", dataset.samples[i]["dataset"], os.path.basename(dataset.samples[i]["path"]))
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    dataset.samples[i]["path"] = img_path
                    Image.fromarray(dataset.samples[i]["img"]).save(img_path)
                    del dataset.samples[i]["img"]

class OCRDataset(GenericDataset):
    """
    Specific class to handle text_recognition/HTR datasets
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.tokenizer = None
        self.reduce_dims_factor = np.array([params["config"]["height_divisor"], params["config"]["width_divisor"], 1])
        self.collate_function = OCRCollateFunction
        self.synthetic_id = 0
        self.synthetic_manager = None

        self.make_syn_val = "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] is not None and "make_val_syn" in self.params["config"]["synthetic_data"] and self.params["config"]["synthetic_data"]["make_val_syn"]
        self.use_syn = (self.set_name == "train" and "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] is not None) \
                        or (self.set_name == "val" and self.make_syn_val)
        self.hist_tokens = {}

    def __getitem__(self, idx: int) -> dict:
        sample = copy.deepcopy(self.samples[idx])

        if not self.load_in_memory:
            sample["img"] = self.get_sample_img(idx)
            sample = apply_preprocessing(sample, self.params["config"]["preprocessing"])

        if self.set_name == "train" and self.use_syn:
            sample = self.synthetic_manager.apply_synthetic(sample, self.training_info)

        # Data augmentation
        if self.set_name == "train" and "data_augmentation" in self.params["config"] and self.params["config"]["data_augmentation"]:
            try:
                sample["img"], sample["applied_da"] = self.apply_data_augmentation(sample["img"])
            except:
                print(sample["img"].shape)
                torch.save({"img": sample["img"], "da": self.params["config"]["data_augmentation"]}, "bug_da.pt")

        sample = OCRDataset.prepare_sample(sample,
                                           set_name=self.set_name,
                                           params=self.params,
                                           reduce_dims_factor=self.reduce_dims_factor,
                                           padding_value=self.padding_value,
                                           include_preprocessing=False,
                                           round_fn=self.params["config"]["round_function"])
        if self.params["config"]["track_hist_tokens"]:
            self.update_hist_tokens(sample)
        return sample

    @staticmethod
    def prepare_sample(sample: dict, set_name: str, params: dict, reduce_dims_factor: tuple[int, int], padding_value: int, include_preprocessing: bool=False, round_fn=np.ceil) -> dict:
        """
        Preprocessing of the sample (in dataloader loop)
        """
        if include_preprocessing:
            sample = apply_preprocessing(sample, params["config"]["preprocessing"])

        if "applied_da" not in sample:
            sample["applied_da"] = None

        if "is_synthetic" not in sample:
            sample["is_synthetic"] = False

        # instance norm requires at least 2 values
        if sample["img"].shape[1] < 2*reduce_dims_factor[1]:
            sample["img"] = pad_image_width_right(sample["img"], 2*reduce_dims_factor[1], padding_value)

        if sample["img"].shape[0] < reduce_dims_factor[0]:
            sample["img"] = pad_image_height_bottom(sample["img"], reduce_dims_factor[0], padding_value)

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = round_fn(sample["img_shape"] / reduce_dims_factor).astype(int)

        # Padding to handle CTC requirements
        if set_name == "train":
            max_label_len = 0
            height = 1
            ctc_padding = False
            if "CTC_line" in params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                ctc_padding = True
            if "CTC_van" in params["config"]["constraints"]:
                max_label_len = max(sample["line_label_len"])
                ctc_padding = True
            if "CTC_span" in params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                height = max(sample["img_reduced_shape"][0], 1)
                ctc_padding = True
            if ctc_padding and 2 * max_label_len + 1 > sample["img_reduced_shape"][1]*height:
                sample["img"] = pad_image_width_right(sample["img"], int(np.ceil((2 * max_label_len + 1) / height) * reduce_dims_factor[1]), padding_value)
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = round_fn(sample["img_shape"] / reduce_dims_factor).astype(int)
            assert all([t >= 1 for t in sample["img_reduced_shape"]])
            sample["img_reduced_shape"] = [t for t in sample["img_reduced_shape"]]

        sample["img_position"] = [[0, sample["img_shape"][0]], [0, sample["img_shape"][1]]]
        # Padding constraints to handle model needs
        if "padding" in params["config"] and params["config"]["padding"]:
            if set_name == "train" or ("train_only" in params["config"]["padding"] and not params["config"]["padding"]["train_only"]):
                if "min_height" in params["config"]["padding"] and "min_width" in params["config"]["padding"]:
                    new_width = max(sample["img"].shape[1], params["config"]["padding"]["min_width"])
                    new_height = max(sample["img"].shape[0], params["config"]["padding"]["min_height"])
                    pad_width = pad_height = None
                else:
                    min_pad = params["config"]["padding"]["min_pad"]
                    max_pad = params["config"]["padding"]["max_pad"]
                    pad_width = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                    pad_height = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                    new_width = params["config"]["padding"]["min_width"]
                    new_height = params["config"]["padding"]["min_height"]

                sample["img"], sample["img_position"] = pad_image(sample["img"], padding_value=padding_value,
                                          new_width=new_width,
                                          new_height=new_height,
                                          pad_width=pad_width,
                                          pad_height=pad_height,
                                          padding_mode=params["config"]["padding"]["mode"],
                                          return_position=True)

        sample["img_reduced_position"] = [np.ceil(p / factor).astype(int) for p, factor in zip(sample["img_position"], reduce_dims_factor[:2])]
        assert sample["img"].dtype == np.uint8
        sample["img"] = np.array(ToTensor()(sample["img"]).permute(1, 2, 0))
        return sample
    
    def update_hist_tokens(self, sample: dict) -> None:
        """
        Track the number of occurrences of each token used for training
        """
        if len(self.hist_tokens) == 0:
            self.hist_tokens = {token_idx: 0 for token_idx in range(self.tokenizer.num_tokens)}
        for token_idx in sample["token_label"]:
            self.hist_tokens[token_idx] += 1

    def get_charset(self):
        """
        Compute the charset considering ground truth of all loaded samples.
        """
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Convert string label to tokens
        """
        for i in range(len(self.samples)):
            self.samples[i] = self.update_sample_labels(self.samples[i])

    def update_sample_labels(self, sample: dict, text: str|None=None) -> dict:
        """
        Convert the label tokens format to match needs of model to be used.
        """
        label = text if text is not None else sample["gt"]["text"]
        if "reverse_gt" in self.params["config"] and self.params["config"]["reverse_gt"]:
            label = list(get_display("".join(label)))
        sample["label"] = label
        sample["token_label"] = self.tokenizer.text_to_token(label)
        if "add_eot" in self.params["config"]["constraints"]:
            sample["token_label"].append(self.tokenizer.tokens["<end>"])
        sample["label_len"] = len(sample["token_label"])
        if "add_sot" in self.params["config"]["constraints"]:
            sample["token_label"].insert(0, self.tokenizer.tokens["<start>"])
        if "fdan_encoding" in self.params["config"]["constraints"]:
            sample["fdan_encoding"] = self.split_tokens_two_step(sample["token_label"])
        elif "mt_fdan_encoding"  in self.params["config"]["constraints"]:
            sample["fdan_encoding"] = self.split_tokens_two_step(sample["token_label"], mt_heads=self.params["num_token_heads"])
        elif "w_dan_encoding" in self.params["config"]["constraints"]:
            token_win = self.params["token_win"]
            for _ in range(token_win):
                sample["token_label"].insert(0, self.tokenizer.tokens["<start>"])
            sample["token_label"].append(self.tokenizer.tokens["<end>"])
            while len(sample["token_label"]) % token_win != 0:
                sample["token_label"].append(self.tokenizer.tokens["<end>"])
            sample["label_len"] = len(sample["token_label"]) - token_win
        elif "mt_dan_encoding" in self.params["config"]["constraints"]:
            num_token_heads = self.params["num_token_heads"]
            sample["token_label"].insert(0, self.tokenizer.tokens["<start>"])
            for _ in range(num_token_heads):
                sample["token_label"].append(self.tokenizer.tokens["<end>"])
            sample["label_len"] = len(sample["token_label"]) - num_token_heads
        elif "meta_dan_encoding" in self.params["config"]["constraints"]:
            num_token_heads = self.params["num_token_heads"]
            token_win = self.params["token_win"]
            for _ in range(token_win):
                sample["token_label"].insert(0, self.tokenizer.tokens["<start>"])
            sample["token_label"].append(self.tokenizer.tokens["<end>"])
            while len(sample["token_label"]) % token_win != 0:
                sample["token_label"].append(self.tokenizer.tokens["<end>"])
            for _ in range(num_token_heads-1):
                sample["token_label"].append(self.tokenizer.tokens["<end>"])
            sample["label_len"] = len(sample["token_label"]) - token_win - num_token_heads + 1


        if "CTC_van" in self.params["config"]["constraints"]:
            sample["line_label"] = list_split(label, "\n")
            sample["token_line_label"] = [self.tokenizer.text_to_token(l) for l in sample["line_label"]]
            sample["line_label_len"] = [len(l) for l in sample["line_label"]]
            sample["nb_lines"] = len(sample["line_label"])
        return sample

    def split_tokens_two_step(self, token_label, mt_heads: int=1) -> dict:
        """
        Split label into two for the two passes of fatser dan
        """
        first_pass_tokens = list()
        lines_tokens = list()
        line_tokens = list()
        linebreak_index = self.tokenizer.tokens["\n"]
        for i in range(len(token_label)):
            if token_label[i] == self.tokenizer.tokens["<pad>"]:
                continue
            # if layout token or fist char of a line
            if token_label[i] not in self.tokenizer.text_indices or (i > 0 and (token_label[i-1] not in self.tokenizer.text_indices or token_label[i-1] == linebreak_index)):
                # add this token for first pass
                if token_label[i] == self.tokenizer.tokens["<end>"]:
                    first_pass_tokens.extend([self.tokenizer.tokens["<end>"], ]* mt_heads)
                else:
                    first_pass_tokens.append(token_label[i])
                # end current line if any and start a new one
                if len(line_tokens) > 0:
                    line_tokens.extend([self.tokenizer.tokens["<end>"], ]* mt_heads)
                    lines_tokens.append(line_tokens)
                    line_tokens = list()
            # if line break, end current line
            elif token_label[i] == linebreak_index:
                line_tokens.extend([self.tokenizer.tokens["<end>"], ]* mt_heads)
                lines_tokens.append(line_tokens)
                line_tokens = list()
                continue
            # if char token, add it to the current line 
            if token_label[i] in self.tokenizer.text_indices:
                line_tokens.append(token_label[i])
        # add last processed line if not already done
        if len(line_tokens) > 0:
            lines_tokens.append(line_tokens)
        mix_passes = lines_tokens.copy()
        mix_passes.insert(0, first_pass_tokens)
        line_indices = np.concatenate([(np.ones(len(mix_passes[i])) * i) for i in range(len(mix_passes))], axis=0)
        index_in_lines = np.concatenate([np.arange(0, len(mix_passes[i])) for i in range(len(mix_passes))], axis=0)
        mix_passes = first_pass_tokens.copy()
        for line in lines_tokens:
            mix_passes.extend(line)
        return {
            "mix_pass": mix_passes,
            "line_indices": line_indices,
            "index_in_lines": index_in_lines,
        }


class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for text_recognition task
    """

    def __init__(self, config):
        self.img_padding_value = config["image_padding_value"]
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        ids = [i for i in range(len(batch_data))]
        formatted_batch_data = {
            "names": names,
            "ids": ids,
        }

        if "token_label" in batch_data[0]:
            labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
            labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
            labels = torch.tensor(labels).long()
            formatted_batch_data["labels"] = labels

        if "label_len" in batch_data[0]:
            formatted_batch_data["labels_len"] = [batch_data[i]["label_len"] for i in range(len(batch_data))]

        if "label" in batch_data[0]:
            formatted_batch_data["raw_labels"] = [batch_data[i]["label"] for i in range(len(batch_data))]

        if "label" in batch_data[0]:
            formatted_batch_data["str_labels"] = ["".join(batch_data[i]["label"]) for i in range(len(batch_data))]

        if "applied_da" in batch_data[0]:
            formatted_batch_data["applied_da"] = [batch_data[i]["applied_da"] for i in range(len(batch_data))]

        if "token_line_label" in batch_data[0]:
            line_labels_len= [batch_data[i]["line_label_len"] for i in range(len(batch_data))]
            token_line_label = [batch_data[i]["token_line_label"] for i in range(len(batch_data))]
            nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]
            pad_line_token = list()
            for i in range(max(nb_lines)):
                current_lines = [token_line_label[j][i] if i < nb_lines[j] else [self.label_padding_value] for j in range(len(batch_data))]
                pad_line_token.append(torch.tensor(pad_sequences_1D(current_lines, padding_value=self.label_padding_value)).long())
                for j in range(len(batch_data)):
                    if i >= nb_lines[j]:
                        line_labels_len[j].append(0)
            line_labels_len = [i for i in zip(*line_labels_len)]
            formatted_batch_data["line_labels_len"] = line_labels_len
            formatted_batch_data["line_labels"] = pad_line_token

        padding_mode = self.config["padding_mode"] if "padding_mode" in self.config else "br"
        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs_position = [batch_data[i]["img_position"] for i in range(len(batch_data))]
        imgs_reduced_position= [batch_data[i]["img_reduced_position"] for i in range(len(batch_data))]
        imgs = pad_images(imgs, padding_value=self.img_padding_value, padding_mode=padding_mode)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)

        formatted_batch_data["imgs"] = imgs
        formatted_batch_data["imgs_position"] = imgs_position
        formatted_batch_data["imgs_reduced_positions"] = imgs_reduced_position
        formatted_batch_data["imgs_shape"] = imgs_shape
        formatted_batch_data["imgs_reduced_shape"] = imgs_reduced_shape

        if "fdan_encoding" in batch_data[0]:
            fdan_labels = [batch_data[i]["fdan_encoding"]["mix_pass"] for i in range(len(batch_data))]
            fdan_labels = pad_sequences_1D(fdan_labels, padding_value=self.label_padding_value)
            fdan_labels = torch.tensor(fdan_labels).long()
            formatted_batch_data["fdan_labels"] = fdan_labels

            fdan_line_indices = [batch_data[i]["fdan_encoding"]["line_indices"] for i in range(len(batch_data))]
            fdan_line_indices = pad_sequences_1D(fdan_line_indices, padding_value=self.label_padding_value)
            fdan_line_indices = torch.tensor(fdan_line_indices).long()
            formatted_batch_data["fdan_line_indices"] = fdan_line_indices

            fdan_index_in_lines = [batch_data[i]["fdan_encoding"]["index_in_lines"] for i in range(len(batch_data))]
            fdan_index_in_lines = pad_sequences_1D(fdan_index_in_lines, padding_value=self.label_padding_value)
            fdan_index_in_lines = torch.tensor(fdan_index_in_lines).long()
            formatted_batch_data["fdan_index_in_lines"] = fdan_index_in_lines

        return formatted_batch_data


