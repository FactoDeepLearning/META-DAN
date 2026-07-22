# -*- coding: utf-8 -*-
"""
Format READ 2016 dataset.
Available at: https://zenodo.org/records/1164045#.YiINkBvjKEA
"""

import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import numpy as np
from lxml import etree
from tqdm import tqdm

from dar.dataset_format.utils import (
    _parse_namespaces,
    clean_text_label,
    load_resize_dpi_save_image,
    resize_image_dpi,
    save_annotations
)

logger = logging.getLogger(__name__)

ORIGINAL_RESOLUTION = 300

SEM_START_TOKENS = {
            "body": "ⓑ",  # paragraph (body)
            "annotation": "ⓐ",  # annotation
            "page": "ⓟ",  # page
            "page-number": "ⓝ",  # page number
            "section": "ⓢ",  # section (=linked annotation + body)
}

SEM_MATCHING_TOKENS = {
            "ⓑ": "Ⓑ",  # paragraph (body)
            "ⓐ": "Ⓐ",  # annotation
            "ⓟ": "Ⓟ",  # page
            "ⓝ": "Ⓝ",  # page number
            "ⓢ": "Ⓢ",  # section (=linked annotation + body)
}


def get_sem_tokens():
    return "".join(list(SEM_MATCHING_TOKENS.values()) + list(SEM_START_TOKENS.values()))


def add_read2016_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "read2016",
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
        default="read2016_page",
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
        choices=["page", "double_page", "paragraph", "line"],
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
        help="Includes layout tokens in ground truth",
    )

    parser.set_defaults(func=run)


def extract_tar_files(data_folder, output_data_folder):
    """
    Check that the dataset files exist and extract them in output_data_folder.
    :param data_folder: Folder containing the dataset zip files.
    :param output_data_folder: Folder containing the uncompressed dataset.
    """
    file_lists = [
        data_folder / path
        for path in [
            "Train-And-Val-ICFHR-2016.tgz",
            "Test-ICFHR-2016.tgz",
        ]
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
    add_sem_tokens,
    level,
    **kwargs
):
    logger.info("Formatting READ 2016 dataset at {} level...".format(level))

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
    elif level == "double_page":
        labels, charset = format_double_page(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
            add_sem_tokens=add_sem_tokens
        )
    elif level == "paragraph":
        labels, charset = format_paragraph(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )
    else:  # level == "line":
        labels, charset = format_line(
            dataset,
            output_path=output_path,
            image_extension=image_extension,
            resolution=resolution,
            add_language_tokens=add_language_tokens,
        )

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary folders
    for set_name in ["train", "val", "test"]:
        shutil.rmtree(output_path / set_name / "Images")
        shutil.rmtree(output_path / set_name / "page")


def format_line(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
):
    """
    Format the READ 2016 dataset at line level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        for page in tqdm(dataset[set_name], desc=f"Formatting {set_name} set"):
            page_img = Image.open(page["img_path"])
            for i, text_region in enumerate(page["text_regions"]):
                for j, line in enumerate(text_region["lines"]):
                    new_img_name = "{}_{}_{}_{}.{}".format(
                        set_name, page["img_path"].stem, i + 1, j + 1, image_extension
                    )
                    new_img_path = output_path / set_name / new_img_name

                    # Crop line region in page image, resize and save
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

                    label = list(line["label"])
                    if add_language_tokens:
                        label.insert(0, "<language-early-modern-german>")
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
    Format the READ 2016 dataset at paragraph level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
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
                    label.insert(0, "<language-early-modern-german>")
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
    add_sem_tokens: bool
):
    """
    Format the READ 2016 dataset at single-page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    :param add_sem_tokens: Whether to add layout otknes or not.
    """
    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        for page in tqdm(dataset[set_name], desc=f"Formatting {set_name} set"):
            new_img_name = "{}_{}.{}".format(
                set_name, page["img_path"].stem, image_extension
            )
            new_img_path = output_path / set_name / new_img_name
            load_resize_dpi_save_image(page["img_path"], new_img_path, ORIGINAL_RESOLUTION, resolution)
            sorted_text_regions, side = sort_text_regions(page["text_regions"], page["width"], add_sem_tokens)

            sep = "" if add_sem_tokens else "\n"
            label = sep.join(text_region["label"] for text_region in sorted_text_regions)
            if add_sem_tokens:
                label = list(SEM_START_TOKENS["page"] + label + SEM_MATCHING_TOKENS[SEM_START_TOKENS["page"]])
            else:
                label = list(label)
            if add_language_tokens:
                label.insert(0, "<language-early-modern-german>")
            label_key = str(Path(output_path.name) / set_name / new_img_name)
            labels[set_name][label_key] = {
                "text": label,
                "side": side
            }
            charset = charset.union(set(label))

    return labels, charset


def format_double_page(
    dataset: dict,
    output_path: Path,
    image_extension: str,
    resolution: int,
    add_language_tokens: bool,
    add_sem_tokens: bool
):
    """
    Format the READ 2016 dataset at double-page level.
    :param dataset: Dict which represents the whole documents for train/val/test.
    :param output_path: Location of the formatted dataset.
    :param resolution: The expected resolution for the formatted images.
    :param image_extension: Format in which formatted images will be saved.
    :param add_language_tokens: Whether to add language tokens in ground truth or not.
    """

    def group_by_page_number(dataset):
        """
        Group pages as couple of successive pages.
        """
        new_dataset = {"train": dict(), "val": dict(), "test": dict()}
        for set_name in ["train", "val", "test"]:
            for page in dataset[set_name]:
                page_num = int(
                    page["text_regions"][0]["label"].replace(".", "").replace(" ", "")
                )
                page["page_num"] = page_num
                if page_num in new_dataset[set_name]:
                    new_dataset[set_name][page_num].append(page)
                else:
                    new_dataset[set_name][page_num] = [
                        page,
                    ]
            new_dataset[set_name] = [
                {
                    "pages": new_dataset[set_name][key],
                    "page_num": new_dataset[set_name][key][0]["page_num"],
                }
                for key in new_dataset[set_name]
            ]
        return new_dataset

    labels = defaultdict(dict)
    charset = set()

    for set_name in ["train", "val", "test"]:
        for i, page in enumerate(dataset[set_name]):
            dataset[set_name][i]["text_regions"], side = sort_text_regions(
                dataset[set_name][i]["text_regions"], dataset[set_name][i]["width"], add_sem_tokens
            )
    dataset = group_by_page_number(dataset)
    for set_name in ["train", "val", "test"]:
        for document in dataset[set_name]:
            # Discard unpaired pages
            if len(document["pages"]) != 2:
                continue
            new_img_name = "{}_{}_{}.{}".format(
                set_name,
                document["pages"][0]["img_path"].stem,
                document["pages"][1]["img_path"].stem,
                image_extension,
            )
            new_img_path = output_path / set_name / new_img_name

            # Merge left and right page images, resize and save
            img_left = np.array(Image.open(document["pages"][0]["img_path"]))
            img_right = np.array(Image.open(document["pages"][1]["img_path"]))
            img = np.concatenate([img_left, img_right], axis=1)
            img = resize_image_dpi(Image.fromarray(img), ORIGINAL_RESOLUTION, resolution)
            img.save(new_img_path)

            label_left = document["pages"][0]["label"]
            label_right = document["pages"][1]["label"]

            if add_sem_tokens:
                page_token = SEM_START_TOKENS["page"]
                end_page_token = SEM_MATCHING_TOKENS[page_token]
                label = list(
                    page_token + label_left + end_page_token + page_token + label_right + end_page_token
                )
            else:
                label = list(
                    label_left + "\n" + label_right
                )
            if add_language_tokens:
                label.insert(0, "<language-early-modern-german>")
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
    # Standardize folder hierarchy
    (path / "PublicData" / "Training").rename(path / "train")
    (path / "PublicData" / "Validation").rename(path / "val")
    (path / "Test-ICFHR-2016").rename(path / "test")
    (path / "PublicData").rmdir()
    (path / "test" / "Images").mkdir(exist_ok=True)
    for filepath in (path / "test").iterdir():
        if filepath.suffix == ".JPG":
            filepath.rename(filepath.parent / "Images" / filepath.name)

    dataset = {
        "train": list(),
        "val": list(),
        "test": list(),
    }
    # XML namespaces
    namespaces = None
    for set_name in ["train", "val", "test"]:
        img_fold_path = path / set_name / "Images"
        xml_fold_path = path / set_name / "page"
        for xml_file_path in sorted(xml_fold_path.iterdir()):
            if xml_file_path.suffix != ".xml":
                continue
            img_path = img_fold_path / Path(xml_file_path.stem).with_suffix(".JPG")
            xml_root = etree.parse(xml_file_path).getroot()

            # Parse the namespace from the content of the XML
            if namespaces is None:
                namespaces = _parse_namespaces(xml_root)

            for page in xml_root.findall("Page", namespaces=namespaces):
                page_dict = {
                    "label": list(),
                    "text_regions": list(),
                    "img_path": img_path,
                    "width": int(page.attrib["imageWidth"]),
                    "height": int(page.attrib["imageHeight"]),
                }
                for text_region in page.findall("TextRegion", namespaces=namespaces):
                    text_region_dict = {
                        "label": list(),
                        "lines": list(),
                        "coords": coord_str_to_points(
                            text_region.find("Coords", namespaces=namespaces).attrib[
                                "points"
                            ]
                        ),
                        "baseline_coords": dict(),
                    }
                    for text_line in text_region.findall(
                        "TextLine", namespaces=namespaces
                    ):
                        label = correct_or_get_text_line_label(
                            text_line, namespaces=namespaces
                        )
                        # Ignore text line with null label
                        if label is None:
                            continue
                        line_baseline = text_line.find(
                            "Baseline", namespaces=namespaces
                        )
                        line_coord = coord_str_to_points(
                            text_line.find("Coords", namespaces=namespaces).attrib[
                                "points"
                            ]
                        )
                        text_line_dict = {
                            "label": label,
                            "coords": line_coord,
                            "baseline_coords": coord_str_to_points(
                                line_baseline.attrib["points"]
                            )
                            if line_baseline is not None
                            else line_coord,
                        }
                        text_line_dict["label"] = text_line_dict["label"]
                        text_region_dict["label"].append(text_line_dict["label"])
                        text_region_dict["lines"].append(text_line_dict)
                    # Ignore text region with null label
                    if text_region_dict["label"] == list():
                        continue
                    text_region_dict["label"] = clean_text_label(
                        "\n".join(text_region_dict["label"])
                    )
                    for side in ["left", "top"]:
                        text_region_dict["baseline_coords"][side] = min(
                            [
                                line["baseline_coords"][side]
                                for line in text_region_dict["lines"]
                            ]
                        )
                    for side in ["bottom", "right"]:
                        text_region_dict["baseline_coords"][side] = max(
                            [
                                line["baseline_coords"][side]
                                for line in text_region_dict["lines"]
                            ]
                        )
                    page_dict["label"].append(text_region_dict["label"])
                    page_dict["text_regions"].append(text_region_dict)
                page_dict["label"] = clean_text_label("\n".join(page_dict["label"]))
                dataset[set_name].append(page_dict)

    return dataset


def correct_or_get_text_line_label(
    xml_text_line: etree._Element, namespaces: Dict[Any, str]
) -> str:
    """
    Get annotation from text line XML entity. Correct it if it is known as a wrong annotation.
    :param text_line: The XML text line entity.
    :return: The string representing the text line annotation.
    """
    if xml_text_line.attrib["id"] == "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2":
        return "16"
    elif xml_text_line.attrib["id"] == "line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477":
        return "108"
    elif xml_text_line.attrib["id"] == "line_e1288df8-8a0d-40df-be91-4b4a332027ec":
        return "196"
    elif xml_text_line.attrib["id"] == "line_455330f3-9e27-4340-ae86-9d6c448dc091":
        return "199"
    elif xml_text_line.attrib["id"] == "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac":
        return "202"
    elif xml_text_line.attrib["id"] == "line_e918616d-64f8-43d2-869c-f687726212be":
        return "214"
    elif xml_text_line.attrib["id"] == "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e":
        return "216"

    text_line_label = (
        xml_text_line.find("TextEquiv", namespaces=namespaces)
        .find("Unicode", namespaces=namespaces)
        .text
    )
    if text_line_label is not None:
        return clean_text_label(text_line_label)
    return None


def coord_str_to_points(coord_str: str) -> Dict[str, int]:
    """
    Extract bounding box from string representing coordinates.
    :param coord_str: String representing coordinates.
    :return: Dict representing bounding box coordinates as top/bottom/left/right.
    """
    points = coord_str.split(" ")
    x_points, y_points = zip(
        *[[int(point.split(",")[0]), int(point.split(",")[1])] for point in points]
    )

    top, bottom, left, right = (
        np.min(y_points),
        np.max(y_points),
        np.min(x_points),
        np.max(x_points),
    )
    return {"left": left, "bottom": bottom, "top": top, "right": right}


def update_label(label: str, add_sem_tokens: bool, start_token: str) -> str:
        """
        Add layout token to text region transcription
        """
        if add_sem_tokens:
            return start_token + label + SEM_MATCHING_TOKENS[start_token]
        return label


def sort_text_regions(text_regions: list, page_width: int, add_sem_tokens: bool = False) -> [list, str]:
    """
    Establish reading order based on paragraph pixel position.
    (Page number, then section by section: first all annotations, then associated body text)
    :param text_regions: List of text regions, each one represented by a dict.
    :param page_width: The width of the page image.
    :param add_sem_tokens: Whether to add layout tokens or not in groundtruth.
    :return: The sorted list of text regions.
    """

    def is_annotation_alone(groups, page_width):
        for group in groups:
            if all(
                [
                    tr["coords"]["right"] < page_width / 2
                    and not tr["label"].replace(".", "").replace(" ", "").isdigit()
                    for tr in group["text_regions"]
                ]
            ):
                return True
        return False

    def merge_group_tr(group, text_region):
        """
        Group text regions togethers, computing global position for them.
        """
        group["text_regions"].append(text_region)

        for coords_type in ["coords", "baseline_coords"]:
            for side in ["top", "left"]:
                group[coords_type][side] = min(
                    [tr[coords_type][side] for tr in group["text_regions"]]
                )
            for side in ["bottom", "right"]:
                group[coords_type][side] = max(
                    [tr[coords_type][side] for tr in group["text_regions"]]
                )

    nb_cols = 1
    groups = list()
    for text_region in text_regions:
        added_in_group = False

        # The page number is always read first.
        temp_label = text_region["label"].replace(".", "").replace(" ", "")
        if len(temp_label) <= 4 and temp_label.isdigit():
            groups.append(
                {
                    "coords": text_region["coords"].copy(),
                    "baseline_coords": text_region["baseline_coords"].copy(),
                    "text_regions": [
                        text_region,
                    ],
                }
            )
            groups[-1]["coords"]["top"] = 0
            groups[-1]["coords"]["bottom"] = 0
            groups[-1]["baseline_coords"]["top"] = 0
            groups[-1]["baseline_coords"]["bottom"] = 0
            continue

        # Merge with previously processed text regions if overlap.
        for group in groups:
            if not (
                group["baseline_coords"]["bottom"]
                <= text_region["baseline_coords"]["top"]
                or group["baseline_coords"]["top"]
                >= text_region["baseline_coords"]["bottom"]
                or (
                    text_region["coords"]["right"] - text_region["coords"]["left"]
                    > 0.4 * page_width
                    and group["coords"]["right"] - group["coords"]["left"]
                    > 0.4 * page_width
                )
            ):
                merge_group_tr(group, text_region)
                added_in_group = True
                break

        # Else consider text region as new group.
        if not added_in_group:
            groups.append(
                {
                    "coords": text_region["coords"].copy(),
                    "baseline_coords": text_region["baseline_coords"].copy(),
                    "text_regions": [
                        text_region,
                    ],
                }
            )

    while is_annotation_alone(groups, page_width):
        new_groups = list()
        annotations_groups = list()
        body_groups = list()
        for i in range(len(groups)):
            group = groups[i]

            # Do not group page number region.
            if all(
                [
                    tr["label"].replace(".", "").replace(" ", "").isdigit()
                    for tr in group["text_regions"]
                ]
            ):
                new_groups.append(group)
                continue

            # Heuristic to detect annotation in margin.
            if all(
                [tr["coords"]["right"] < page_width / 2 for tr in group["text_regions"]]
            ):
                annotations_groups.append(group)
                continue
            body_groups.append(group)

        # For a given section, annotations are read from top to bottom, before the main body text.
        for ag in annotations_groups:
            dist = [
                min(
                    [
                        abs(ag["coords"]["bottom"] - g["coords"]["top"]),
                        abs(ag["coords"]["top"] - g["coords"]["bottom"]),
                    ]
                )
                for g in body_groups
            ]
            index = np.argmin(dist)
            for tr in ag["text_regions"]:
                merge_group_tr(body_groups[index], tr)

        new_groups.extend(body_groups)
        groups = new_groups
    ordered_groups = sorted(groups, key=lambda g: g["coords"]["top"])
    sorted_text_regions = list()
    for group in ordered_groups:
        text_regions = group["text_regions"]
        # Page number region.
        if (
            len(text_regions) == 1
            and text_regions[0]["label"].replace(".", "").replace(" ", "").isdigit()
        ):
            side = "right" if text_regions[0]["coords"]["left"] > page_width / 2 else "left"
            sorted_text_regions.append(text_regions[0])
            sorted_text_regions[-1]["mode"] = "page_number"
            sorted_text_regions[-1]["label"] = update_label(sorted_text_regions[-1]["label"], add_sem_tokens, SEM_START_TOKENS["page-number"])
        else:
            # Separate text regions as belonging to left or right page, then order by vertical axis.
            left = [
                tr
                for tr in group["text_regions"]
                if tr["coords"]["right"] < page_width / 2
            ]
            right = [
                tr
                for tr in group["text_regions"]
                if tr["coords"]["right"] >= page_width / 2
            ]
            nb_cols = max(2 if len(left) > 0 else 1, nb_cols)
            for i, text_region in enumerate(
                sorted(left, key=lambda tr: tr["coords"]["top"])
            ):
                sorted_text_regions.append(text_region)
                sorted_text_regions[-1]["mode"] = "annotation"
                sorted_text_regions[-1]["label"] = update_label(sorted_text_regions[-1]["label"], add_sem_tokens, SEM_START_TOKENS["annotation"])
                if i==0 and add_sem_tokens:
                    sorted_text_regions[-1]["label"] = SEM_START_TOKENS["section"] + sorted_text_regions[-1]["label"]

            for i, text_region in enumerate(
                sorted(right, key=lambda tr: tr["coords"]["top"])
            ):
                sorted_text_regions.append(text_region)
                sorted_text_regions[-1]["mode"] = "body"
                sorted_text_regions[-1]["label"] = update_label(sorted_text_regions[-1]["label"], add_sem_tokens, SEM_START_TOKENS["body"])
                if i == 0 and add_sem_tokens and len(left) == 0:
                    sorted_text_regions[-1]["label"] = SEM_START_TOKENS["section"] + sorted_text_regions[-1]["label"]
                if i == len(right) - 1 and add_sem_tokens:
                    sorted_text_regions[-1]["label"] = sorted_text_regions[-1]["label"] + SEM_MATCHING_TOKENS[SEM_START_TOKENS["section"]]

    return sorted_text_regions, side
