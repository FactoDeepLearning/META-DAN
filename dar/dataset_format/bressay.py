# -*- coding: utf-8 -*-
"""
Format BRESSAY dataset.
Available at: https://tc11.cvc.uab.es/datasets/BRESSAY_1
"""

import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import os
import shutil
from PIL import Image

from dar.dataset_format.utils import (
    clean_text_label,
    load_resize_dpi_save_image,
    natural_sort,
    save_annotations
)

logger = logging.getLogger(__name__)


def add_bressay_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "bressay",
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
        default="bressay_page",
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
            "paragraph",
            "line"
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
    filepath = data_folder / "bressay.zip"
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
    logger.info("Formatting BRESSAY dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    temp_fold_path = output_path / "Temp"
    extract_zip_files(data_folder, temp_fold_path)

    set_files = get_set_files(output_path)

    if level == "page":
        labels, charset = format_page(
            set_files,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )
    elif level == "paragraph":
        labels, charset = format_paragraph(
            set_files,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )
    # level == "line"
    else:
        labels, charset = format_line(
            set_files,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )


    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    shutil.rmtree(temp_fold_path)


def format_page(
    set_files: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the BRESSAY dataset at single-page level.
    :param set_files: Dict which represents the list of document ids for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()
    page_folder = output_path / "Temp" / "bressay" / "data" / "pages"

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for i, page_id in enumerate(tqdm(set_files[set_name], desc=f"Formatting {set_name} set")):
            img_path = page_folder / page_id / f"{page_id}.png"
            label_path = page_folder / page_id / f"{page_id}.txt"
            with open(label_path, "r") as f:
                label = clean_text_label(handle_strikethrough(f.read()))
                label = list(label)

            new_img_name = "{}_{}.{}".format(
                set_name, img_path.stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name

            # We consider approximation of A4 format
            image = Image.open(img_path)
            source_dpi = image.width / 8.267

            load_resize_dpi_save_image(img_path, new_img_path, source_dpi, resolution)

            if add_language_tokens:
                label.insert(0, "<language-portuguese>")

            label_key = str(Path(output_path.name) / set_name / new_img_name)
            labels[set_name][label_key] = {
                "text": label,
            }
            charset = charset.union(set(label))

    return labels, charset


def format_paragraph(
    set_files: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the BRESSAY dataset at paragraph level.
    :param set_files: Dict which represents the list of document ids for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()
    paragraph_folder = output_path / "Temp" / "bressay" / "data" / "paragraphs"

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for i, page_id in enumerate(tqdm(set_files[set_name], desc=f"Formatting {set_name} set")):
            # We consider approximation of A4 format
            page_img_path = output_path / "Temp" / "bressay" / "data" / "pages" / page_id / f"{page_id}.png"
            image = Image.open(page_img_path)
            source_dpi = image.width / 8.267

            page_folder = paragraph_folder / page_id
            img_paths = [page_folder / img_name for img_name in natural_sort(os.listdir(page_folder)) if ".png" in img_name]
            for j, img_path in enumerate(img_paths):
                label_path = img_path.parent / img_path.name.replace(".png", ".txt")
                with open(label_path, "r") as f:
                    label = clean_text_label(handle_strikethrough(f.read()))
                    label = list(label)

                new_img_name = "{}_{}.{}".format(
                    set_name, img_path.stem, image_extension
                )
                new_img_path = output_path / set_name / new_img_name

                load_resize_dpi_save_image(img_path, new_img_path, source_dpi, resolution)

                if add_language_tokens:
                    label.insert(0, "<language-portuguese>")

                label_key = str(Path(output_path.name) / set_name / new_img_name)
                labels[set_name][label_key] = {
                    "text": label,
                }
                charset = charset.union(set(label))

    return labels, charset


def format_line(
    set_files: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the BRESSAY dataset at line level.
    :param set_files: Dict which represents the list of document ids for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()
    paragraph_folder = output_path / "Temp" / "bressay" / "data" / "lines"

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for i, page_id in enumerate(tqdm(set_files[set_name], desc=f"Formatting {set_name} set")):
            # We consider approximation of A4 format
            page_img_path = output_path / "Temp" / "bressay" / "data" / "pages" / page_id / f"{page_id}.png"
            image = Image.open(page_img_path)
            source_dpi = image.width / 8.267

            page_folder = paragraph_folder / page_id
            img_paths = [page_folder / img_name for img_name in natural_sort(os.listdir(page_folder)) if ".png" in img_name]
            for img_path in img_paths:
                label_path = img_path.parent / img_path.name.replace(".png", ".txt")
                with open(label_path, "r") as f:
                    label = clean_text_label(handle_strikethrough(f.read()))
                    label = list(label)

                if len(label) <= 0:
                    print(f"ignoring {img_path}: no text label")
                    continue

                new_img_name = "{}_{}.{}".format(
                    set_name, img_path.stem, image_extension
                )
                new_img_path = output_path / set_name / new_img_name

                load_resize_dpi_save_image(img_path, new_img_path, source_dpi, resolution)

                if add_language_tokens:
                    label.insert(0, "<language-portuguese>")

                label_key = str(Path(output_path.name) / set_name / new_img_name)
                labels[set_name][label_key] = {
                    "text": label,
                }
                charset = charset.union(set(label))

    return labels, charset


def get_set_files(path: Path) -> Dict:
    """
    Get the list of files for training/validation/test.
    :param path: Path for the formatted dataset.
    :return: Dict with set_name as key and list of document ids as value.
    """

    set_files = {
        "train": "training.txt",
        "val": "validation.txt",
        "test": "test.txt"
    }

    for set_name in ["train", "val", "test"]:
        set_file_path = path / "Temp" / "bressay" / "sets" / set_files[set_name]
        with open(set_file_path, "r") as f:
            set_files[set_name] = [l.strip() for l in f.readlines()]

    return set_files


def handle_strikethrough(label):
    """
    Strikethrough text is ignored. Only readable subscript/superscript is preserved.
    """
    for pattern in ["##@@???@@##", "$$@@???@@$$", "@@???@@", "##--xxx--##", "$$--xxx--$$", "--xxx--",]:
        label = label.replace(pattern, "")

    for pattern in ["##--([^#-$]+?)--##", "$$--([^#-$]+?)--$$", "--([^#-$]+?)--"]:
        label = re.sub(pattern, "", label, flags=re.DOTALL)

    for pattern in ["##([^#-$]+?)##", "$$([^#-$]+?)$$", ]:
        label = re.sub(pattern, r"\1", label, flags=re.DOTALL)
    return label


