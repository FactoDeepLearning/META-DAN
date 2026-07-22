"""
Module handling Datasets in a generic way.
"""

from __future__ import annotations
import torch
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dar.generic.transforms import apply_data_augmentation
from dar.generic.utils import natural_sort
from dar.generic.dataset_sampler import MultiDatasetSampler
import os
import numpy as np
from PIL import Image
import cv2
import json


class DatasetManager:
    """
    Main class to handle all datasets (train, val, and test) sampling and loading
    """

    def __init__(self, params: dict):
        self.params = params
        self.dataset_class = params["dataset_class"]
        self.img_padding_value = params["config"]["image_padding_value"]

        self.my_collate_function = None

        self.train_dataset = None
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_loader = None
        self.valid_loaders = dict()
        self.test_loaders = dict()

        self.train_sampler = None
        self.valid_samplers = dict()
        self.test_samplers = dict()

        self.generator = torch.Generator()
        self.generator.manual_seed(0)

        self.batch_size = {
            "train": self.params["train_batch_size"],
            "val": self.params["val_batch_size"],
            "test": self.params["test_batch_size"],
        }

    def apply_specific_treatment_after_dataset_loading(self, dataset: GenericDataset) -> None:
        raise NotImplementedError

    def load_datasets(self) -> None:
        """
        Load training and validation datasets
        """
        self.train_dataset = self.dataset_class(self.params, "train", "train", self.get_paths_and_sets(self.params["train"]))

        self.my_collate_function = self.train_dataset.collate_function(self.params["config"])
        self.apply_specific_treatment_after_dataset_loading(self.train_dataset)

        for val_set in self.params["val"]:
            self.valid_datasets[val_set["name"]] = self.dataset_class(self.params, "val", val_set["name"], self.get_paths_and_sets(val_set["sets"]))
            self.apply_specific_treatment_after_dataset_loading(self.valid_datasets[val_set["name"]])

    def load_samplers(self) -> None:
        """
        Load training and validation data samplers
        """
        if self.params["use_ddp"]:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=True)
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = DistributedSampler(self.valid_datasets[custom_name], num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=False)
        else:
            if self.params["config"]["dataset_same_sampler"]:
                self.train_sampler = MultiDatasetSampler(self.train_dataset)
            for custom_name in self.valid_datasets.keys():
                self.valid_samplers[custom_name] = None

    def load_dataloaders(self) -> None:
        """
        Load training and validation data loaders
        """
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size["train"],
                                       shuffle=True if self.train_sampler is None else False,
                                       drop_last=False,
                                       batch_sampler=self.train_sampler if isinstance(self.train_sampler, BatchSampler) else None,
                                       sampler=self.train_sampler if isinstance(self.train_sampler, DistributedSampler) or isinstance(self.train_sampler, RandomSampler) else None,
                                       num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                       pin_memory=True,
                                       collate_fn=self.my_collate_function,
                                       worker_init_fn=self.seed_worker,
                                       generator=self.generator)

        for key in self.valid_datasets.keys():
            self.valid_loaders[key] = DataLoader(self.valid_datasets[key],
                                                 batch_size=self.batch_size["val"],
                                                 sampler=self.valid_samplers[key] if isinstance(self.valid_samplers[key], DistributedSampler) else None,
                                                 batch_sampler=self.valid_samplers[key] if isinstance(self.valid_samplers[key], BatchSampler) else None,
                                                 shuffle=False,
                                                 num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=self.my_collate_function,
                                                 worker_init_fn=self.seed_worker,
                                                 generator=self.generator)

    @staticmethod
    def seed_worker(worker_id) -> None:
        """
        Set seed for each worker
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def generate_test_loader(self, set_name: str) -> None:
        """
        Load test dataset, data sampler and data loader
        """
        test_set = [test_set for test_set in self.params["test"] if test_set["name"] == set_name][0]
        self.test_datasets[test_set["name"]] = self.dataset_class(self.params, "test", test_set["name"], self.get_paths_and_sets(test_set["sets"]))
        self.apply_specific_treatment_after_dataset_loading(self.test_datasets[test_set["name"]])
        if self.params["use_ddp"]:
            self.test_samplers[test_set["name"]] = DistributedSampler(self.test_datasets[test_set["name"]], num_replicas=self.params["num_gpu"], rank=self.params["ddp_rank"], shuffle=False)
        else:
            self.test_samplers[test_set["name"]] = None
        self.test_loaders[test_set["name"]] = DataLoader(self.test_datasets[test_set["name"]],
                                                    batch_size=self.batch_size["test"],
                                                    sampler=self.test_samplers[test_set["name"]] if isinstance(self.test_samplers[test_set["name"]] , DistributedSampler) else None,
                                                    batch_sampler=self.test_samplers[test_set["name"]] if isinstance(self.test_samplers[test_set["name"]] , BatchSampler) else None,
                                                    shuffle=False,
                                                    num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    collate_fn=self.my_collate_function,
                                                    worker_init_fn=self.seed_worker,
                                                    generator=self.generator)

    def remove_test_dataset(self, custom_name: str) -> None:
        """
        Remove test dataset, sampler et loader by dict key
        """
        del self.test_datasets[custom_name]
        del self.test_samplers[custom_name]
        del self.test_loaders[custom_name]

    def remove_valid_dataset(self, custom_name: str) -> None:
        """
        Remove val dataset, sampler et loader by dict key
        """
        del self.valid_datasets[custom_name]
        del self.valid_samplers[custom_name]
        del self.valid_loaders[custom_name]

    def remove_train_dataset(self) -> None:
        """
        Remove train dataset, sampler et loader
        """
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None

    def remove_all_datasets(self) -> None:
        """
        Remove all train/val/test datasets, samplers and loaders
        """
        self.remove_train_dataset()
        for name in list(self.valid_datasets.keys()):
            self.remove_valid_dataset(name)
        for name in list(self.test_datasets.keys()):
            self.remove_test_dataset(name)

    def get_paths_and_sets(self, datasets: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Format the datasets and split to use by replacing dataset name by dataset path.
        Single-dataset example:
        Input
        [
            {
                "dataset_name": "rimes2009_page,
                "split": "train"
            }
        ]
        Output
        [
            {
                "path": "./Datasets/formatted/rimes2009_page",
                "set_name": "train"
            }
        ]
        """
        paths_and_sets = list()
        for current_set in datasets:
            path = os.path.join(self.params["dataset_folder"], current_set["dataset_name"])
            paths_and_sets.append({
                "path": path,
                "set_name": current_set["split"]
            })
        return paths_and_sets


class GenericDataset(Dataset):
    """
    Main class to handle single dataset
    """

    def __init__(self, params: dict, set_name: str, custom_name: str, paths_and_sets: list[dict]):
        self.params = params
        self.name = custom_name
        self.set_name = set_name
        self.mean = np.array(params["config"]["mean"]) if "mean" in params["config"].keys() else None
        self.std = np.array(params["config"]["std"]) if "std" in params["config"].keys() else None

        self.load_in_memory = self.params["config"]["load_in_memory"] if "load_in_memory" in self.params["config"] else True

        self.samples = self.load_samples(paths_and_sets, load_in_memory=self.load_in_memory)

        if self.load_in_memory:
            self.apply_preprocessing(params["config"]["preprocessing"])

        self.padding_value = params["config"]["image_padding_value"]

        self.curriculum_config = None
        self.training_info = {}

    def __len__(self):
        """
        Compute the number of sample in the dataset
        """
        return len(self.samples)

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load an image by path as a numpy array of shape (H, W, C)
        """
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
            ## grayscale images
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img

    @staticmethod
    def load_samples(paths_and_sets: list[dict], load_in_memory: bool=True) -> list[dict]:
        """
        Load labels of all samples (and images if load_in_memory is True)
        """
        samples = list()
        for path_and_set in paths_and_sets:
            path = path_and_set["path"]
            set_name = path_and_set["set_name"]
            with open(os.path.join(path, "labels.json"), "rb") as f:
                labels = json.load(f)
                gt = labels[set_name]
                for filename in natural_sort(gt.keys()):
                    name = os.path.join(os.path.dirname(path), filename)
                    full_path = name
                    samples.append({
                        "name": name,
                        "gt": gt[filename],
                        "path": full_path,
                        "dataset": os.path.basename(path)
                    })
                    if load_in_memory:
                        samples[-1]["img"] = GenericDataset.load_image(full_path)
        return samples

    def apply_preprocessing(self, preprocessing: list[dict])-> None:
        """
        Apply each preprocessing on all samples.
        Each preprocessing is represented by a dict stating its kind and configuration.
        """
        for i in range(len(self.samples)):
            self.samples[i] = apply_preprocessing(self.samples[i], preprocessing)

    def compute_std_mean(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulated variance and mean of whole dataset.
        Mean and variance are computed per channels.
        """
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        if not self.load_in_memory:
            sample = self.samples[0].copy()
            sample["img"] = self.get_sample_img(0)
            img = apply_preprocessing(sample, self.params["config"]["preprocessing"])["img"]
        else:
            img = self.get_sample_img(0)
        _, _, c = img.shape
        sum = np.zeros((c,))
        nb_pixels = 0

        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessing"])["img"]
            else:
                img = self.get_sample_img(i)
            sum += np.sum(img, axis=(0, 1))
            nb_pixels += np.prod(img.shape[:2])
        mean = sum / nb_pixels
        diff = np.zeros((c,))
        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessing"])["img"]
            else:
                img = self.get_sample_img(i)
            diff += [np.sum((img[:, :, k] - mean[k]) ** 2) for k in range(c)]
        std = np.sqrt(diff / nb_pixels)

        self.mean = mean
        self.std = std
        return mean, std

    def apply_data_augmentation(self, img: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Apply data augmentation strategy on the input numpy array image.
        Return a tuple containing:
            - the updated image as a numpy array
            - a list of applied data augmentation technique names
        """
        return apply_data_augmentation(img, self.params["config"]["data_augmentation"])

    def get_sample_img(self, i: int) -> np.ndarray:
        """
        Get image by index as a numpy array of shape (H, W, C)
        """
        if self.load_in_memory:
            return self.samples[i]["img"]
        else:
            return GenericDataset.load_image(self.samples[i]["path"])

    def denormalize(self, img: np.array) -> np.array:
        """
        Get original image, before normalization
        """
        return img * self.std + self.mean


def apply_preprocessing(sample: dict, preprocessing: list[dict]) -> dict:
    """
    Update the given sample by applying each preprocessing defined in the given list of preprocessing.
    """
    resize_ratio = [1, 1]
    img = sample["img"]
    for preprocessing in preprocessing:

        if preprocessing["type"] == "dpi":
            ratio = preprocessing["target"] / preprocessing["source"]
            temp_img = img
            h, w, c = temp_img.shape
            temp_img = cv2.resize(temp_img, (int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img

            resize_ratio = [ratio, ratio]

        if preprocessing["type"] == "to_grayscaled":
            temp_img = img
            h, w, c = temp_img.shape
            if c == 3:
                img = np.expand_dims(
                    0.2125 * temp_img[:, :, 0] + 0.7154 * temp_img[:, :, 1] + 0.0721 * temp_img[:, :, 2],
                    axis=2).astype(np.uint8)

        if preprocessing["type"] == "to_RGB":
            temp_img = img
            h, w, c = temp_img.shape
            if c == 1:
                img = np.concatenate([temp_img, temp_img, temp_img], axis=2)

        if preprocessing["type"] == "resize":
            keep_ratio = preprocessing["keep_ratio"]
            max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]
            temp_img = img
            h, w, c = temp_img.shape

            ratio_h = max_h / h if max_h else 1
            ratio_w = max_w / w if max_w else 1
            if keep_ratio:
                ratio_h = ratio_w = min(ratio_w, ratio_h)
            new_h = min(max_h, int(h * ratio_h))
            new_w = min(max_w, int(w * ratio_w))
            temp_img = cv2.resize(temp_img, (new_w, new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)

            img = temp_img
            resize_ratio = [ratio_h, ratio_w]

        if preprocessing["type"] == "fixed_height":
            new_h = preprocessing["height"]
            temp_img = img
            h, w, c = temp_img.shape
            ratio = new_h / h
            temp_img = cv2.resize(temp_img, (int(w*ratio), new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img
            resize_ratio = [ratio, ratio]
    if resize_ratio != [1, 1] and "raw_line_seg_label" in sample:
        for li in range(len(sample["raw_line_seg_label"])):
            for side, ratio in zip((["bottom", "top"], ["right", "left"]), resize_ratio):
                for s in side:
                    sample["raw_line_seg_label"][li][s] = sample["raw_line_seg_label"][li][s] * ratio

    sample["img"] = img
    sample["resize_ratio"] = resize_ratio
    return sample

