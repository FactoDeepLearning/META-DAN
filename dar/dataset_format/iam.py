# -*- coding: utf-8 -*-
"""
Format IAM dataset.
Available at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
"""

import json
import logging
import shutil
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict
from PIL import Image
from tqdm import tqdm

from dar.dataset_format.utils import (
    clean_text_label,
    load_resize_dpi_save_image,
    resize_image_dpi,
    save_annotations
)

logger = logging.getLogger(__name__)

SPLIT_FILE = Path("dar", "dataset_format", "resources", "iam_split.json")
ORIGINAL_RESOLUTION = 300

def add_iam_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "iam",
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
        default="iam_page",
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
        choices=["page", "paragraph", "line", "word"],
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


def extract_tar_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset tar files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """
    file_lists = [
        data_folder / path
        for path in ["formsA-D.tgz", "formsE-H.tgz", "formsI-Z.tgz", "xml.tgz"]
    ]
    for filepath in file_lists:
        if not filepath.exists():
            logger.error(f"{filepath} not found")
            exit(-1)
        tar = tarfile.open(filepath)
        tar.extractall(output_data_folder)
        tar.close()


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
    logger.info("Formatting IAM dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    extract_tar_files(data_folder, output_path / "temp")

    dataset = preformat_dataset(output_path)

    if level == "page":
        format_function = format_page
    elif level == "paragraph":
        format_function = format_paragraph
    elif level == "line":
        format_function = format_line
    else:  # level == "word":
        format_function = format_word

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
    shutil.rmtree(output_path / "temp")


def format_word(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the IAM dataset at word level.
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
            page_img = Image.open(page["img_path"])
            for i, line in enumerate(page["lines"]):
                for j, word in enumerate(line["words"]):
                    new_img_name = "{}_{}_{}_{}.{}".format(
                        set_name, page["img_path"].stem, i, j, image_extension
                    )
                    new_img_path = output_path / set_name / new_img_name

                    # discard words without segmentation annotation
                    if word["coord"] is None:
                        continue

                    word_img = page_img.copy().crop(
                        (
                            word["coord"]["left"],
                            word["coord"]["top"],
                            word["coord"]["right"],
                            word["coord"]["bottom"],
                        )
                    )
                    word_img = resize_image_dpi(word_img, ORIGINAL_RESOLUTION, resolution)
                    word_img.save(new_img_path)
                    label = list(word["label"])
                    if add_language_tokens:
                        label.insert(0, "<language-english>")
                    label_key = str(Path(output_path.name) / set_name / new_img_name)
                    labels[set_name][label_key] = {
                        "text": label,
                    }
                    charset = charset.union(set(label))

    return labels, charset


def format_line(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the IAM dataset at line level.
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
            page_img = Image.open(page["img_path"])
            for i, line in enumerate(page["lines"]):
                new_img_name = "{}_{}_{}.{}".format(
                    set_name, page["img_path"].stem, i, image_extension
                )
                new_img_path = output_path / set_name / new_img_name
                line_img = page_img.copy().crop(
                    (
                        line["coord"]["left"],
                        line["coord"]["top"],
                        line["coord"]["right"],
                        line["coord"]["bottom"],
                    )
                )
                line_img = resize_image_dpi(line_img, ORIGINAL_RESOLUTION, resolution)
                line_img.save(new_img_path)
                label = list(line["label"])
                if add_language_tokens:
                    label.insert(0, "<language-english>")
                label_key = str(Path(output_path.name) / set_name / new_img_name)
                labels[set_name][label_key] = {
                    "text": label,
                }
                charset = charset.union(set(label))

    return labels, charset


def format_paragraph(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the IAM dataset at paragraph level.
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
            new_img_name = "{}_{}.{}".format(
                set_name, page["img_path"].stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name
            page_img = Image.open(page["img_path"])
            paragraph_img = page_img.copy().crop(
                (
                    page["coord"]["left"],
                    page["coord"]["top"],
                    page["coord"]["right"],
                    page["coord"]["bottom"],
                )
            )
            paragraph_img = resize_image_dpi(paragraph_img, ORIGINAL_RESOLUTION, resolution)
            paragraph_img.save(new_img_path)
            label = list(page["label"])
            if add_language_tokens:
                label.insert(0, "<language-english>")
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
    Format the IAM dataset at page level.
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
        for page in dataset[set_name]:
            new_img_name = "{}_{}.{}".format(
                set_name, page["img_path"].stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name
            load_resize_dpi_save_image(page["img_path"], new_img_path, ORIGINAL_RESOLUTION, resolution)
            label = (
                list(page["printed_label"])
                + [
                    "\n",
                ]
                + list(page["label"])
            )
            if add_language_tokens:
                label.insert(0, "<language-english>")
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

    with SPLIT_FILE.open("r") as f:
        split = json.load(f)

    for set_name in ["train", "val", "test"]:
        filenames = split[set_name]
        for filename in filenames:
            img_path = path / "temp" / f"{filename}.png"
            xml_path = path / "temp" / f"{filename}.xml"
            xml_root = ET.parse(xml_path).getroot()
            printed_label = "\n".join(
                [
                    text_line.attrib["text"]
                    for text_line in xml_root.find("machine-printed-part").findall(
                        "machine-print-line"
                    )
                ]
            )
            lines = list()
            for line in xml_root.find("handwritten-part").findall("line"):
                words = list()
                for word in line.findall("word"):
                    bottom = list()
                    left = list()
                    right = list()
                    top = list()
                    cmps = word.findall("cmp")
                    if len(cmps) == 0:
                        coord = None
                    else:
                        for cmp in cmps:
                            left.append(int(cmp.attrib["x"]))
                            right.append(
                                int(cmp.attrib["x"]) + int(cmp.attrib["width"])
                            )
                            top.append(int(cmp.attrib["y"]))
                            bottom.append(
                                int(cmp.attrib["y"]) + int(cmp.attrib["height"])
                            )
                        coord = {
                            "bottom": max(bottom),
                            "top": min(top),
                            "left": min(left),
                            "right": max(right),
                        }

                    words.append(
                        {"label": clean_text_label(word.attrib["text"]), "coord": coord}
                    )

                lines.append(
                    {
                        "label": clean_text_label(line.attrib["text"]),
                        "coord": {
                            "bottom": max(
                                [w["coord"]["bottom"] for w in words if w["coord"]]
                            ),
                            "top": min(
                                [w["coord"]["top"] for w in words if w["coord"]]
                            ),
                            "left": min(
                                [w["coord"]["left"] for w in words if w["coord"]]
                            ),
                            "right": max(
                                [w["coord"]["right"] for w in words if w["coord"]]
                            ),
                        },
                        "words": words,
                    }
                )
            dataset[set_name].append(
                {
                    "img_path": img_path,
                    "label": clean_text_label(
                        "\n".join([line["label"] for line in lines])
                    ),
                    "printed_label": printed_label,
                    "coord": {
                        "bottom": max([line["coord"]["bottom"] for line in lines]),
                        "top": min([line["coord"]["top"] for line in lines]),
                        "left": min([line["coord"]["left"] for line in lines]),
                        "right": max([line["coord"]["right"] for line in lines]),
                    },
                    "lines": lines,
                }
            )
    return dataset
