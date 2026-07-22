# -*- coding: utf-8 -*-
"""
Format ScribbleLens dataset.
Available at: https://openslr.org/84/
"""

import logging
import shutil
import stat
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm

from dar.dataset_format.utils import load_resize_dpi_save_image, natural_sort, save_annotations

logger = logging.getLogger(__name__)


def add_scribblelens_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "scribblelens",
        description=__doc__,
        help=__doc__,
    )
    parser.add_argument(
        "-d",
        "--data-folder",
        type=Path,
        default=Path("data"),
        help="Path to the directory containing raw data",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=Path("formatted_data"),
        help="Path to the directory that will contain the formatted data",
    )
    parser.add_argument(
        "-n",
        "--output-dataset-name",
        type=str,
        default="scribblelens_page",
        help="Name of the formatted dataset",
    )
    parser.add_argument(
        "-e",
        "--image-extension",
        type=str,
        default="jpg",
        choices=["png", "jpg"],
        help="Format in which formatted images will be saved",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=str,
        default="page",
        choices=[
            "page",
            "line",
        ],
        help="Level in which to format the dataset",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=150,
        help="Fixed resolution (in dpi) of the images",
    )
    parser.add_argument(
        "--add-language-tokens",
        action="store_true",
        help="Include language tokens in ground truth",
    )

    parser.set_defaults(func=run)


def extract_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset tar files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """
    file_lists = [
        data_folder / path
        for path in [
            "scribblelens.corpus.v1.2.zip",
            "scribblelens.supplement.original.pages.tgz",
        ]
    ]
    for filepath in file_lists:
        if not filepath.exists():
            logger.error(f"{filepath} not found")
            exit(-1)
        if filepath.suffix == ".tgz":
            tar = tarfile.open(filepath)
            tar.extractall(output_data_folder)
            tar.close()
        elif filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath) as zip_file:
                zip_file.extractall(output_data_folder)


def run(
    data_folder,
    output_folder,
    output_dataset_name,
    image_extension,
    resolution,
    add_language_tokens,
    level,
    **kwargs
):
    logger.info("Formatting ScribbleLens dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    extract_files(data_folder, output_path)

    dataset = preformat_dataset(output_path)

    if level == "page":
        format_function = format_page
    else:  # level == "line"
        format_function = format_line

    labels, charset = format_function(
        dataset,
        output_path=output_path,
        image_extension=image_extension,
        resolution=resolution,
        add_language_tokens=add_language_tokens,
    )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    shutil.rmtree(output_path / "scribblelens.corpus.v1")


def format_line(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the ScribbleLens dataset at line level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for page in tqdm(dataset[set_name], desc=f"Formatting {set_name} set"):
            page["img_path"].chmod(stat.S_IRWXU)

            # This is an assumption
            original_dpi = 150 if "kieft" in str(page["img_path"]) else 300

            for i, line in enumerate(page["lines"]):
                new_img_name = "{}_{}_{}.{}".format(
                    set_name, page["img_path"].stem, i, image_extension
                )

                load_resize_dpi_save_image(
                    line["img_path"],
                    output_path / set_name / new_img_name,
                    original_dpi,
                    resolution,
                )
                label = list(line["label"])
                if add_language_tokens:
                    label.insert(0, "<language-dutch>")
                label_key = str(Path(output_path.name) / set_name / new_img_name)
                labels[set_name][label_key] = {
                    "text": label,
                }
                charset = charset.union(set(label))

    return labels, charset


def format_page(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the ScribbleLens dataset at page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for page in tqdm(dataset[set_name], desc=f"Formatting {set_name} set"):
            page["img_path"].chmod(stat.S_IRWXU)
            new_img_name = "{}_{}.{}".format(
                set_name, page["img_path"].stem, image_extension
            )

            # This is an assumption
            original_dpi = 150 if "kieft" in str(page["img_path"]) else 300

            load_resize_dpi_save_image(
                page["img_path"],
                output_path / set_name / new_img_name,
                original_dpi,
                resolution,
            )
            label = list(page["label"])
            if add_language_tokens:
                label.insert(0, "<language-dutch>")
            label_key = str(Path(output_path.name) / set_name / new_img_name)
            labels[set_name][label_key] = {
                "text": label,
            }
            charset = charset.union(set(label))

    return labels, charset


def preformat_dataset(path: Path) -> Dict:
    """
    Preformat the annotation of the whole dataset.
    :param path: Path for the formatted dataset.
    :return: Dict with representation of whole documents for train/val/test.
    """
    dataset = {
        "train": list(),
        "val": list(),
        "test": list(),
    }

    # Create validation set by taking 10% of examples from training set, for each subset
    split_folder_path = path / "scribblelens.corpus.v1" / "corpora"
    split = {
        "train": list(),
        "val": list(),
        "test": [
            Path(filepath)
            for filepath in (split_folder_path / "nl.pages.test.txt")
            .read_text()
            .strip()
            .split("\n")
        ],
    }

    train_val_img_paths = (
        (split_folder_path / "nl.pages.train.txt").read_text().strip().split("\n")
    )

    path_by_subset = dict()
    for filepath in train_val_img_paths:
        subset_name = filepath.split("/")[1]
        if subset_name not in path_by_subset.keys():
            path_by_subset[subset_name] = list()
        path_by_subset[subset_name].append(filepath)

    for key in path_by_subset.keys():
        paths = [Path(filepath) for filepath in natural_sort(path_by_subset[key])]
        train_len = int(np.ceil(len(paths) * 0.9))
        split["train"].extend(paths[:train_len])
        split["val"].extend(paths[train_len:])

    for set_name in ["train", "val", "test"]:
        for filepath in split[set_name]:
            folder_path = path / "scribblelens.corpus.v1" / filepath.parent
            lines = list()
            line_paths = natural_sort(
                [
                    str(line_path)
                    for line_path in folder_path.iterdir()
                    if "line.color" in str(line_path)
                    and "wholepage" not in str(line_path)
                ]
            )
            line_paths = [Path(line_path) for line_path in line_paths]
            for i, line_path in enumerate(line_paths):
                line_label = (line_path.parent / f"line{i}.txt").read_text().strip()

                lines.append({"label": line_label, "img_path": line_path})

            page_label = "\n".join(line["label"] for line in lines)
            dataset[set_name].append(
                {
                    "img_path": [
                        filepath
                        for filepath in folder_path.iterdir()
                        if filepath.stem.startswith("original")
                    ][0],
                    "label": page_label,
                    "lines": lines,
                }
            )
    return dataset
