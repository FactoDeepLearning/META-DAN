# -*- coding: utf-8 -*-
"""
Format Esposalles dataset.
"""

import logging
import os
from PIL import Image
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict
from zipfile import ZipFile
from tqdm import tqdm
from lxml import etree

from dar.dataset_format.utils import (
    _parse_namespaces,
    load_resize_dpi_save_image,
    natural_sort,
    resize_image_dpi,
    save_annotations,
    extract_page_transcription
)

logger = logging.getLogger(__name__)

FILENAME = "IEHHR-XMLpages"
ORIGINAL_RESOLUTION = 300

def add_esposalles_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "esposalles",
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
        default="esposalles_page",
        help="Name of the formatted dataset",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=str,
        default="page",
        choices=["page", "paragraph", "line"],
        help="Level in which to format the dataset",
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
        "-r",
        "--resolution",
        type=int,
        default=150,
        help="Fixed resolution (in dpi) of the images",
    )
    parser.add_argument(
        "--exclude-margin",
        action="store_true",
        help="Exclude margin from image (for page level only)",
    )
    parser.add_argument(
        "--add-language-tokens",
        action="store_true",
        help="Include language tokens in ground truth",
    )

    parser.set_defaults(func=run)


def extract_zip_files(data_folder, tmp_data_folder):
    """
    Check that the dataset files exist and extract them in tmp_data_folder.
    :param data_folder: Folder containing the dataset zip files.
    :param tmp_data_folder: Temporary folder containing the unzipped dataset.
    """
    file_lists = [
        data_folder / path
        for path in [
            Path(FILENAME).with_suffix(".zip"),
            Path(FILENAME + "_test_with_GT").with_suffix(".zip"),
        ]
    ]
    for filepath in file_lists:
        if not filepath.exists():
            logger.error(f"{filepath} not found")
            exit(-1)
        with ZipFile(filepath) as zip_file:
            zip_file.extractall(tmp_data_folder)

def run(
    data_folder,
    output_folder,
    output_dataset_name,
    level,
    image_extension,
    resolution,
    exclude_margin,
    add_language_tokens,
    **kwargs
):
    logger.info("Formatting Esposalles dataset...")

    # Extract data
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    temp_fold_path = output_path / "temp"
    temp_fold_path.mkdir(parents=True, exist_ok=True)
    extract_zip_files(data_folder, temp_fold_path)

    dataset = preformat_dataset(output_path)

    if level == "page":
        labels, charset = format_page(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            exclude_margin=exclude_margin,
            add_language_tokens=add_language_tokens
        )
    elif level == "paragraph":
        labels, charset = format_paragraph(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )
    else:  # level == "line"
        labels, charset = format_line(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folder
    shutil.rmtree(temp_fold_path)


def format_line(
        dataset: dict,
        output_path: Path,
        image_extension: str,
        resolution: int,
        add_language_tokens: bool,
):
    """
    Format the Esposalles dataset at line level.
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
        for i, page in enumerate(tqdm(dataset[set_name])):
            page_img = Image.open(page["img_path"])
            for j, text_region in enumerate(page["text_regions"]):
                for k, line in enumerate(text_region["lines"]):
                    new_img_name = "{}_{}_{}_{}.{}".format(
                        set_name, Path(page["img_path"]).stem, j+1, k+1, image_extension
                    )
                    new_img_path = output_path / set_name / new_img_name
                    line_img = page_img.copy().crop(
                        (
                            line["coords"]["left"],
                            line["coords"]["top"],
                            line["coords"]["right"],
                            line["coords"]["bottom"],
                        )
                    )
                    line_img = resize_image_dpi(line_img, ORIGINAL_RESOLUTION, resolution)
                    line_img.save(new_img_path)

                    label = list(line["text"])
                    if add_language_tokens:
                        label.insert(0, "<language-spanish>")
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
    Format the Esposalles dataset at paragraph level.
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
        for i, page in enumerate(tqdm(dataset[set_name])):
            page_img = Image.open(page["img_path"])
            for j, text_region in enumerate(page["text_regions"]):
                new_img_name = "{}_{}_{}.{}".format(
                    set_name, Path(page["img_path"]).stem, j+1, image_extension
                )
                new_img_path = output_path / set_name / new_img_name
                region_img = page_img.copy().crop(
                    (
                        text_region["coords"]["left"],
                        text_region["coords"]["top"],
                        text_region["coords"]["right"],
                        text_region["coords"]["bottom"],
                    )
                )
                region_img = resize_image_dpi(region_img, ORIGINAL_RESOLUTION, resolution)
                region_img.save(new_img_path)

                label = list(text_region["text"])
                if add_language_tokens:
                    label.insert(0, "<language-spanish>")
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
    exclude_margin: bool,
    add_language_tokens: bool,
):
    """
    Format the Esposalles dataset at page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param exclude_margin: Exclude margin from page image.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for i, page in enumerate(tqdm(dataset[set_name])):
            new_img_name = "{}_{}.{}".format(
                set_name, Path(page["img_path"]).stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name
            if exclude_margin:
                page_img = Image.open(page["img_path"])
                page_img = page_img.copy().crop(
                    (
                        page["coords"]["left"],
                        page["coords"]["top"],
                        page["coords"]["right"],
                        page["coords"]["bottom"],
                    )
                )
                page_img = resize_image_dpi(page_img, ORIGINAL_RESOLUTION, resolution)
                page_img.save(new_img_path)
            else:
                load_resize_dpi_save_image(page["img_path"], new_img_path, ORIGINAL_RESOLUTION, resolution)
            label = list(page["text"])
            if add_language_tokens:
                label.insert(0, "<language-spanish>")
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

    train_fold = path / "temp" / "train"
    train_val_paths = natural_sort([str(f) for f in train_fold.glob("*.xml")])

    test_fold = path / "temp" / "test"
    test_gt_fold = path / "temp" / "esposalles_test_with_gt"
    for filename in os.listdir(test_gt_fold):
        shutil.move(os.path.join(test_gt_fold, filename), os.path.join(test_fold, filename))
    test_paths = [str(p) for p in test_fold.glob("*.xml")]

    file_paths = {
        "train": train_val_paths[:75],
        "val": train_val_paths[75:],
        "test": test_paths
    }

    # XML namespaces
    namespaces = None

    # Create output image directories
    for set_name in ["train", "val", "test"]:
        for xml_path in file_paths[set_name]:
            img_path = xml_path.replace(".xml", ".jpg")
            xml_page_anns = etree.parse(xml_path).getroot()

            # Parse the namespace from the content of the XML
            if namespaces is None:
                namespaces = _parse_namespaces(xml_page_anns)

            # Save page_transcription and update charset
            page_dict = extract_page_transcription(xml_page_anns, namespaces)
            page_dict["img_path"] = img_path

            dataset[set_name].append(page_dict)

    return dataset
