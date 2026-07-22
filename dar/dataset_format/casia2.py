# -*- coding: utf-8 -*-
"""
Format CASIA2.0-2.2 dataset.
Available at: https://nlpr.ia.ac.cn/databases/handwriting/Download.html
"""


import logging
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
import os
import struct
from PIL import Image
from tqdm import tqdm
import numpy as np

from dar.dataset_format.utils import (
    natural_sort,
    resize_image_dpi,
    save_annotations
)

logger = logging.getLogger(__name__)

ORIGINAL_RESOLUTION = 300

def add_casia2_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "casia2",
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
        default="casia2_page",
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


def extract_zip_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset zip files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """

    files = {
        "train": ["HWDB2.0Train.zip", "HWDB2.1Train.zip", "HWDB2.2Train.zip"],
        "test": ["HWDB2.0Test.zip", "HWDB2.1Test.zip", "HWDB2.2Test.zip"]
    }

    for set_name in ["train", "test"]:
        for filename in files[set_name]:
            filepath = data_folder / filename
            if not filepath.exists():
                logger.error(f"{filepath} not found")
                exit(-1)
            with zipfile.ZipFile(filepath) as zip_file:
                zip_file.extractall(output_data_folder / set_name)


def load_dgrl_file(file_path):
    #  Adapted from https://github.com/peterWon/CASIA-HWDB-Recognition/blob/master/processHWDB.py
    page_dict = {
        "text": list(),
        "lines": list()
    }
    with open(file_path, "rb") as f:
        header_size = struct.unpack("I", f.read(4))[0]

        illustration_len = header_size - 36
        code_format = f.read(8)
        illus = f.read(illustration_len)
        code_type = str(f.read(20), 'ASCII')
        code_length = struct.unpack("h", f.read(2))[0]
        bits_per_pixel = struct.unpack("h", f.read(2))[0]

        height = struct.unpack("I", f.read(4))[0]
        width = struct.unpack("I", f.read(4))[0]
        num_lines = struct.unpack("I", f.read(4))[0]
        image = np.zeros(shape=[height, width], dtype=np.uint8) + 255

        garbage = bytes([255 for _ in range(code_length)])
        line_images = list()
        for i in range(num_lines):
            num_chars = struct.unpack("I", f.read(4))[0]
            label = [f.read(code_length) for _ in range(num_chars)]
            label = [str(code, ('gbk'))[0] for code in label if code != garbage]

            top = struct.unpack("I", f.read(4))[0]
            left = struct.unpack("I", f.read(4))[0]
            char_height = struct.unpack("I", f.read(4))[0]
            char_width = struct.unpack("I", f.read(4))[0]
            right = left + char_width
            bottom = top + char_height

            line_image = np.fromfile(f, dtype='uint8', count=char_height * char_width).reshape(char_height, char_width)
            line_images.append(line_image)
            stroke_mask = line_image != 255
            image[top:bottom, left:right][stroke_mask] = line_image[stroke_mask]
            page_dict["lines"].append({
                "text": label,
                "coords": {
                    "top": top,
                    "bottom": bottom,
                    "right": right,
                    "left": left
                }
            })
    page_dict["text"] = list("\n".join(["".join(l["text"]) for l in page_dict["lines"]]))

    return image, line_images, page_dict


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
    logger.info("Formatting CASIA2 dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    temp_path = output_path / "temp"
    temp_path.mkdir(parents=True, exist_ok=True)
    extract_zip_files(data_folder, temp_path)


    if level == "page":
        labels, charset = format_page(
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )
    else:  # level == "line"
        labels, charset = format_line(
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    shutil.rmtree(temp_path)


def format_line(
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the CASIA2 dataset at line level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """

    train_images = natural_sort([str(output_path / "temp" / "train" /filename) for filename in os.listdir(os.path.join(output_path, "temp", "train"))])
    test_images = natural_sort([str(output_path / "temp" / "test" / filename) for filename in os.listdir(os.path.join(output_path, "temp", "test"))])
    val_images = train_images[-475:]
    train_images = train_images[:-475]

    labels = defaultdict(dict)
    charset = set()

    for set_name, image_paths in zip(
        ["train", "val", "test"], [train_images, val_images, test_images]
    ):
        (output_path / set_name).mkdir(exist_ok=True)
        for image_path in tqdm(image_paths, desc=f"Formatting {set_name} set"):

            try:
                page_image, line_images, label = load_dgrl_file(image_path)
            except:
                print(f"Ignoring {image_path}: error reading file")
                continue

            for j, line_image in enumerate(line_images):
                line_label = label["lines"][j]
                new_img_name = os.path.basename(image_path).replace(".dgrl", f"_{j}.{image_extension}")
                new_img_path = output_path / set_name / new_img_name
                line_text = line_label["text"]

                if len(line_text) <= 0:
                    print(f"Ignoring {image_path}, line {j}")
                    continue

                line_image = page_image.copy()[line_label["coords"]["top"]:line_label["coords"]["bottom"], line_label["coords"]["left"]:line_label["coords"]["right"]]
                line_image = Image.fromarray(line_image)
                line_image = resize_image_dpi(line_image, ORIGINAL_RESOLUTION, resolution)
                line_image.save(new_img_path)

                if add_language_tokens:
                    line_text.insert(0, "<language-chinese>")
                label_key = str(Path(output_path.name) / set_name / new_img_name)
                labels[set_name][label_key] = {
                    "text": line_text,
                }
                charset = charset.union(set(line_text))

    return labels, charset


def format_page(
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the CASIA2 dataset at single-page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """

    train_images = natural_sort([str(output_path / "temp" / "train" /filename) for filename in os.listdir(os.path.join(output_path, "temp", "train"))])
    test_images = natural_sort([str(output_path / "temp" / "test" / filename) for filename in os.listdir(os.path.join(output_path, "temp", "test"))])
    val_images = train_images[-475:]
    train_images = train_images[:-475]

    labels = defaultdict(dict)
    charset = set()

    for set_name, image_paths in zip(
        ["train", "val", "test"], [train_images, val_images, test_images]
    ):
        (output_path / set_name).mkdir(exist_ok=True)
        for image_path in tqdm(image_paths, desc=f"Formatting {set_name} set"):

            try:
                page_image, line_images, label = load_dgrl_file(image_path)
            except:
                print(f"Ignoring {image_path}: error reading file")
                continue
            new_img_name = os.path.basename(image_path).replace(".dgrl", f".{image_extension}")
            new_img_path = output_path / set_name / new_img_name

            image = Image.fromarray(page_image)
            image = resize_image_dpi(image, ORIGINAL_RESOLUTION, resolution)
            image.save(new_img_path)
            label = label["text"]
            if add_language_tokens:
                label.insert(0, "<language-chinese>")
            label_key = str(Path(output_path.name) / set_name / new_img_name)
            labels[set_name][label_key] = {
                "text": label,
            }
            charset = charset.union(set(label))

    return labels, charset





