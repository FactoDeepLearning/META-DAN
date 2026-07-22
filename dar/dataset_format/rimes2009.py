# -*- coding: utf-8 -*-
"""
Format RIMES 2009 dataset.
"""

import logging
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict
from PIL import Image
from tqdm import tqdm

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


def add_rimes2009_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "rimes2009",
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
        default="rimes2009_page",
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


def extract_tar_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset tar files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """
    filepath = data_folder / "RIMES_page.tar.gz"
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
    add_sem_tokens,
    **kwargs
):
    logger.info("Formatting RIMES 2009 dataset at {} level...".format(level))

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    extract_tar_files(data_folder, output_path)

    dataset = preformat_dataset(output_path)

    if level == "page":
        labels, charset = format_page(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
            add_sem_tokens=add_sem_tokens
        )
    else:  # level == "paragraph"
        labels, charset = format_paragraph(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    shutil.rmtree(output_path / "RIMES page")


def format_paragraph(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the RIMES 2009 dataset at paragraph level.
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
            for i, text_region in enumerate(page["text_regions"]):
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
    add_language_tokens: bool,
    add_sem_tokens: bool,
):
    """
    Format the RIMES 2009 dataset at single-page level.
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

    img_folder_path = path / "RIMES page" / "Images"
    xml_folder_path = path / "RIMES page" / "XML"
    xml_files = natural_sort([str(path) for path in xml_folder_path.iterdir()])
    train_xml = xml_files[:1050]
    valid_xml = xml_files[1050:1150]
    test_xml = xml_files[1150:]
    for set_name, xml_files in zip(
        ["train", "val", "test"], [train_xml, valid_xml, test_xml]
    ):
        for i, xml_path in enumerate(xml_files):
            text_regions = list()
            root = ET.parse(xml_path).getroot()
            img_name = root.find("source").text

            # Add missing text region annotation.
            if img_name == "01160_L.png":
                text_regions.append(
                    {
                        "label": "LETTRE RECOMMANDEE\nAVEC ACCUSE DE RECEPTION",
                        "type": "",
                        "coords": {
                            "left": 88,
                            "right": 1364,
                            "top": 1224,
                            "bottom": 1448,
                        },
                    }
                )

            for text_region in root.findall("box"):
                type = text_region.find("type").text
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
                    "img_path": img_folder_path / img_name,
                    "label": "\n".join([tr["label"] for tr in text_regions]),
                    "sem_label": "".join([update_label_sem(tr["label"], tr["type"]) for tr in text_regions]),
                }
            )
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

