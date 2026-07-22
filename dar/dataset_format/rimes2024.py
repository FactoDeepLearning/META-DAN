# -*- coding: utf-8 -*-
"""
Format RIMES 2024 dataset.
Available at: https://zenodo.org/records/10812725
"""

import json
import logging
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

from dar.dataset_format.utils import (
    clean_text_label,
    load_resize_dpi_save_image,
    natural_sort,
    resize_image_dpi,
    save_annotations
)

logger = logging.getLogger(__name__)

ORIGINAL_RESOLUTION = 300

SEM_START_TOKENS = {
            'Ouverture': "ⓞ",  # opening
            'Corps de texte': "ⓑ",  # body
            'PS/PJ': "ⓟ",  # post scriptum
            'Coordonnées Expéditeur': "ⓢ",  # sender
            'Reference': "ⓢ",  # also counted as sender information
            'Objet': "ⓨ",  # why
            'Date, Lieu': "ⓦ",  # where, when
            'Coordonnées Destinataire': "ⓡ",  # recipient
}

SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",
            "ⓞ": "Ⓞ",
            "ⓡ": "Ⓡ",
            "ⓢ": "Ⓢ",
            "ⓦ": "Ⓦ",
            "ⓨ": "Ⓨ",
            "ⓟ": "Ⓟ"
}


def get_sem_tokens():
    return "".join(list(SEM_MATCHING_TOKENS.values()) + list(SEM_START_TOKENS.values()))


def add_rimes2024_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "rimes2024",
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
        default="rimes2024_page",
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
        ],
        help="Level in which to format the dataset",
    )
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        default=["L", ],
        choices=["L", "F"], # Q not implemented
        help="Filter sub-datasets from RIMES2024 depending on categories",
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
    parser.add_argument(
        "--add-sem-tokens",
        action="store_true",
        help="Include semantic tokens in ground truth",
    )

    parser.set_defaults(func=run)


def extract_zip_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset zip files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """
    filepath = data_folder / "Images_Courriers.zip"
    if not filepath.exists():
        logger.error(f"{filepath} not found")
        exit(-1)
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
    categories,
    add_sem_tokens,
    **kwargs
):
    logger.info("Formatting RIMES 2024 dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    extract_zip_files(data_folder, output_path)

    dataset = preformat_dataset(output_path)

    if level == "page":
        labels, charset = format_page(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            categories=categories,
            add_language_tokens=add_language_tokens,
            add_sem_tokens=add_sem_tokens
        )
    else:  # level == "paragraph"
        labels, charset = format_paragraph(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            categories=categories,
            add_language_tokens=add_language_tokens,
        )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    shutil.rmtree(output_path / "Images_Courriers")


def format_paragraph(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    categories: list[str],
    add_language_tokens: bool,
):
    """
    Format the RIMES 2024 dataset at paragraph level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param categories: Filter dataset by given categories.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for page in dataset[set_name]:
            if page["category"] not in categories:
                continue
            page_img = Image.open(page["img_path"])
            for i, text_region in enumerate(page["text_regions"]):
                if text_region["type"] != "Corps de texte":
                    continue
                new_img_name = "{}_{}_{}.{}".format(
                    set_name, page["img_path"].stem, i + 1, image_extension
                )
                new_img_path = output_path / set_name / new_img_name

                # Crop paragraph in page image, resize and save
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

                label = list(text_region["label"])
                if add_language_tokens:
                    label.insert(0, "<language-french>")
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
    categories: list[str],
    add_language_tokens: bool,
    add_sem_tokens: bool,
):
    """
    Format the RIMES 2024 dataset at single-page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param categories: Filter dataset by given categories.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    :param add_sem_tokens: Whether to add semantic tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        (output_path / set_name).mkdir(exist_ok=True)
        for i, page in enumerate(tqdm(dataset[set_name], desc=f"Formatting {set_name} set")):
            if page["category"] not in categories:
                continue
            new_img_name = "{}_{}.{}".format(
                set_name, page["img_path"].stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name
            load_resize_dpi_save_image(page["img_path"], new_img_path, ORIGINAL_RESOLUTION, resolution)
            label = list(page["sem_label"] if add_sem_tokens else page["label"])
            if add_language_tokens:
                label.insert(0, "<language-french>")
            label_key = str(Path(output_path.name) / set_name / new_img_name)
            labels[set_name][label_key] = {
                "text": label,
            }
            charset = charset.union(set(label))

    return labels, charset


def split_dataset():
    page_ids = [f'{n:05}' for n in range(1, 5606)]
    with open(Path("dar", "dataset_format", "resources", "rimes2011_page_id.json")) as f:
        rimes2011split = json.load(f)
    for set_name in ["train", "val", "test"]:
        for page_id in rimes2011split[set_name]:
            del page_ids[page_ids.index(page_id)]
    new_split = rimes2011split
    while len(new_split["train"]) < 4500:
        new_split["train"].append(page_ids[0])
        del page_ids[0]
    while len(new_split["val"]) < 505:
        new_split["val"].append(page_ids[0])
        del page_ids[0]
    while len(new_split["test"]) < 600:
        new_split["test"].append(page_ids[0])
        del page_ids[0]
    return new_split


def preformat_dataset(path: Path) -> Dict:
    """
    Preformat the annotation of the whole dataset.
    :param path: Path for the formatted dataset.
    :return: Dict with representation of whole documents for train/val/test.
    """
    split = split_dataset()
    main_folder = path / "Images_Courriers"

    image_names = list()
    for i in range(1, 4):
        folder = f"DVD{i}_TIF"
        image_names.extend([str(main_folder / folder / img_name) for img_name in os.listdir(main_folder / folder) if ".tif" in img_name])

    image_names = natural_sort(image_names)

    dataset = {
        "train": list(),
        "val": list(),
        "test": list(),
    }

    train_images = [im for im in image_names if im.split("/")[-1].split("_")[0] in split["train"]]
    val_images = [im for im in image_names if im.split("/")[-1].split("_")[0] in split["val"]]
    test_images = [im for im in image_names if im.split("/")[-1].split("_")[0] in split["test"]]
    for set_name, image_paths in zip(
        ["train", "val", "test"], [train_images, val_images, test_images]
    ):
        for i, image_path in enumerate(image_paths):
            text_regions = list()
            xml_path = image_path.replace(".tif", ".xml")
            root = ET.parse(xml_path).getroot()
            img_name = root.find("source").text
            category = image_path.split(".")[0][-1]

            for text_region in root.findall("box"):
                type = text_region.find("type").text
                type = type.replace("Coordonnes Destinataire", "Coordonnées Destinataire")
                label = text_region.find("text").text
                if label is None or len(label.strip()) <= 0:
                    continue

                label = format_label(label, type, img_name)

                coords = {
                    "left": int(text_region.attrib["top_left_x"]),
                    "right": int(text_region.attrib["bottom_right_x"]),
                    "top": int(text_region.attrib["top_left_y"]),
                    "bottom": int(text_region.attrib["bottom_right_y"]),
                }
                text_regions.append({"label": label, "type": type, "coords": coords})

            text_regions = order_text_regions(text_regions)

            dataset[set_name].append(
                {
                    "text_regions": text_regions,
                    "img_path": Path(image_path),
                    "label": "\n".join([tr["label"] for tr in text_regions]),
                    "category": category
                }
            )
            if category == "L":
                dataset[set_name][-1]["sem_label"] = "".join([update_label_sem(tr["label"], tr["type"]) for tr in text_regions])
    return dataset


def update_label_sem(label: str, tr_type: str) -> str:
    """
    Add layout tokens
    """
    if tr_type == "":
        return label
    begin_token = SEM_START_TOKENS[tr_type]
    end_token = SEM_MATCHING_TOKENS[begin_token]
    return begin_token + label + end_token


def format_label(label: str, type: str, img_name: str) -> str:
    """
    Format and correct ground truth annotation.
    :param label: The string annotation.
    :param type: The semantic of the text.
    :param img_name: The name of the corresponding image.
    :return: The formatted fixed annotation.
    """
    # fix incorrect annotations
    if label == "Ref : QVLCP¨65":
        label = label.replace("¨", "")
    if img_name == "01094_L.png" and type == "Corps de texte":
        label = "Suite à la tempête du 19.11.06, un\narbre est tombé sur mon toît et l'a endommagé.\nJe d'eplore une cinquantaine de tuiles à changer,\nune poutre à réparer et une gouttière à\nremplacer. Veuillez trouver ci-joint le devis\nde réparation. Merci de m'envoyer votre\nexpert le plus rapidement possible.\nEn esperant une réponse rapide de votre\npart, veuillez accepter, madame, monsieur,\nmes salutations distinguées."
    elif img_name == "01111_L.png" and type == "Corps de texte":
        label = "Je vous ai envoyé un courrier le 20 octobre 2006\nvous signalant un sinistre survenu dars ma\nmaison, un dégât des eaux consécutif aux\nfortes pluis.\nVous deviez envoyer un expert pour constater\nles dégâts. Personne n'est venu à ce jour\nJe vous prie donc de faire le nécessaire\nafin que les réparations nécessaires puissent\nêtre commencés.\nDans l'attente, veuillez agréer, Monsieur,\nmes sincères salutations"

    label = fix_label_accent(label)
    label = fix_label_choice(label)
    label = clean_text_label(label)
    return label


def fix_label_choice(label: str) -> str:
    """
    Some annotations present many options for a given text part, always keep the first one only.
    :param label: The ground truth string annotation.
    :return: The corrected annotation.
    """
    if "¤" in label:
        label = re.sub("¤{([^¤]*)[/|]([^¤]*)}¤", r"\1", label, flags=re.DOTALL)
        label = re.sub(
            "¤{([^¤]*)[/|]([^¤]*)[/|]([^¤]*)>", r"\1", label, flags=re.DOTALL
        )
        label = re.sub("¤([^¤]*)[/|]([^¤]*)¤", r"\1", label, flags=re.DOTALL)
        label = re.sub("¤{}¤([^¤]*)[/|]([^ ]*)", r"\1", label, flags=re.DOTALL)
        label = re.sub("¤{/([^¤]*)/([^ ]*)", r"\1", label, flags=re.DOTALL)
        label = re.sub("¤{([^¤]*)[/|]([^ ]*)", r"\1", label, flags=re.DOTALL)
        label = re.sub("([^¤]*)/(.*)[¤}{]+", r"\1", label, flags=re.DOTALL)
        label = re.sub("[¤}{]+([^¤}{]*)[¤}{]+", r"\1", label, flags=re.DOTALL)
        label = re.sub("¤([^¤]*)¤", r"\1", label, flags=re.DOTALL)
    label = re.sub("[ ]+", " ", label, flags=re.DOTALL)
    label = label.strip()
    return label


def fix_label_accent(label: str) -> str:
    """
    Solve encoding issues for accented characters.
    :param label: The ground truth string annotation.
    :return: The corrected annotation.
    """
    return (
        label.replace("\\n", "\n")
        .replace("<euro>", "€")
        .replace(">euro>", "€")
        .replace(">fligne>", " ")
        .replace("Â¤", "¤")
        .replace("Ã»", "û")
        .replace("�", "")
        .replace("ï¿©", "é")
        .replace("Ã§", "ç")
        .replace("Ã©", "é")
        .replace("Ã´", "ô")
        .replace("\xa0", " ")
        .replace("Ã¨", "è")
        .replace("Â°", "°")
        .replace("Ã", "À")
        .replace("Ã¬", "À")
        .replace("Ãª", "ê")
        .replace("Ã®", "î")
        .replace("Ã¢", "â")
        .replace("Â²", "²")
        .replace("Ã¹", "ù")
        .replace("Ã", "à")
        .replace("¬", "€")
    )


def order_text_regions(text_regions: list) -> list:
    """
    Sort text region to have a coherent reading order, based on their positions.
    :param text_regions: List of text region representations.
    :return: The sorted list of text region representations.
    """
    sorted_text_regions = list()
    for tr in text_regions:
        added = False
        # Init list with first text region.
        if len(sorted_text_regions) == 0:
            sorted_text_regions.append(tr)
            added = True
        # Check where to place current text region with respect to the others, based on rules.
        else:
            for i, sorted_tr in enumerate(sorted_text_regions):
                tr_height = tr["coords"]["bottom"] - tr["coords"]["top"]
                sorted_tr_height = (
                    sorted_tr["coords"]["bottom"] - sorted_tr["coords"]["top"]
                )
                tr_is_totally_above = (
                    tr["coords"]["bottom"] < sorted_tr["coords"]["top"]
                )
                tr_is_top_above = tr["coords"]["top"] < sorted_tr["coords"]["top"]
                is_same_level = (
                    sorted_tr["coords"]["top"]
                    <= tr["coords"]["bottom"]
                    <= sorted_tr["coords"]["bottom"]
                    or sorted_tr["coords"]["top"]
                    <= tr["coords"]["top"]
                    <= sorted_tr["coords"]["bottom"]
                    or tr["coords"]["top"]
                    <= sorted_tr["coords"]["bottom"]
                    <= tr["coords"]["bottom"]
                    or tr["coords"]["top"]
                    <= sorted_tr["coords"]["top"]
                    <= tr["coords"]["bottom"]
                )
                vertical_shared_space = (
                    tr["coords"]["bottom"] - sorted_tr["coords"]["top"]
                    if tr_is_top_above
                    else sorted_tr["coords"]["bottom"] - tr["coords"]["top"]
                )
                reach_same_level_limit = vertical_shared_space > 0.3 * min(
                    tr_height, sorted_tr_height
                )
                is_more_at_left = tr["coords"]["left"] < sorted_tr["coords"]["left"]
                equivalent_height = abs(tr_height - sorted_tr_height) < 0.3 * min(
                    tr_height, sorted_tr_height
                )
                is_middle_above_top = (
                    np.mean([tr["coords"]["top"], tr["coords"]["bottom"]])
                    < sorted_tr["coords"]["top"]
                )
                if (
                    tr_is_totally_above
                    or (
                        is_same_level
                        and equivalent_height
                        and is_more_at_left
                        and reach_same_level_limit
                    )
                    or (
                        is_same_level
                        and equivalent_height
                        and tr_is_top_above
                        and not reach_same_level_limit
                    )
                    or (is_same_level and not equivalent_height and is_middle_above_top)
                ):
                    sorted_text_regions.insert(i, tr)
                    added = True
                    break
        if not added:
            sorted_text_regions.append(tr)

    return sorted_text_regions

