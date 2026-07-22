# -*- coding: utf-8 -*-
"""
Format MAURDOR dataset.
Available at: https://catalogue.elra.info/en-us/repository/browse/ELRA-E0045/
"""

import logging
import re
import shutil
import tarfile
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from PIL import Image
from dar.dataset_format.utils import save_annotations, clean_text_label
from tqdm import tqdm

logger = logging.getLogger(__name__)


def add_maurdor_formatter_parser(subcommands):
    parser = subcommands.add_parser(
        "maurdor",
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
        default="maurdor_page",
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
        "-r",
        "--resolution",
        type=int,
        default=150,
        help="Fixed resolution (in dpi) of the images",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=["french", "english", "arabic"],
        choices=["french", "english", "arabic", "other"],
        help="Filter sub-datasets from MAURDOR depending on languages",
    )
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        default=["C3", "C4"],
        choices=["C3", "C4"],
        help="Filter sub-datasets from MAURDOR depending on categories",
    )
    parser.add_argument(
        "-s",
        "--scripts",
        nargs="+",
        default=[
            "hand",
            "typed",
        ],
        help="Filter sub-datasets from MAURDOR depending on script mode",
    )
    parser.add_argument(
        "--add-language-tokens",
        action="store_true",
        help="Include language tokens in ground truth",
    )

    parser.set_defaults(func=run)


def extract_tar_files(data_folder, output_path):
    """
    Check that the dataset files exist and extract them in extract_folder.
    :param data_folder: Folder containing the dataset zip files.
    :param output_path: Folder containing the uncompressed dataset.
    """
    print("Extracting files...")
    file_lists = [
        Path(path)
        for path in [
            "train/XML.tar.gz",
            "train/TIFF.tar.gz",
            "valid/XML.tar.gz",
            "valid/TIFF.tar.gz",
            "test/XML.tar.gz",
            "test/TIFF.tar.gz",
        ]
    ]
    for filename in file_lists:
        filepath = data_folder / filename
        if not filepath.exists():
            logger.error(f"{filepath} not found")
            exit(-1)
        extract_folder = output_path / "temp" / filename.parent
        tar = tarfile.open(filepath)
        tar.extractall(extract_folder)
        tar.close()


def run(
    data_folder,
    output_folder,
    output_dataset_name,
    image_extension,
    resolution,
    languages,
    categories,
    scripts,
    add_language_tokens,
    **kwargs
):
    logger.info("Formatting MAURDOR dataset...")

    # Check dataset files and extract
    output_path = output_folder / output_dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    extract_tar_files(data_folder, output_path)

    dataset = preformat_dataset(
        output_path, add_language_tokens=add_language_tokens
    )
    dataset = filter_dataset(
        dataset,
        languages=languages,
        categories=categories,
        scripts=scripts,
        filter_orientation=True,
        a4_only=True,
    )

    labels = defaultdict(dict)
    charset = set()
    for set_name in dataset.keys():
        i = 0
        for doc in tqdm(dataset[set_name], desc=f"Formatting {set_name} set"):
            img_path = doc["img_path"]
            for page_id, page in enumerate(doc["pages"]):
                pil_img = Image.open(img_path)
                pil_img.seek(page_id)
                img = np.array(pil_img)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                img = rotate_image(img, page["orientation"])
                page, img = change_a4_page_img_resolution(page, img, dpi=resolution)
                new_img_name = f"{img_path.stem}_{page_id}.{image_extension}"
                new_img_path = output_path / set_name / new_img_name
                Image.fromarray(img).save(new_img_path)
                label_key = str(Path(output_path.name) / set_name / new_img_name)
                label = list(clean_text_label("".join(page["label"])))
                labels[set_name][label_key] = {
                    "text": label,
                }
                charset = charset.union(set(page["label"]))
                i += 1

    # dict key: "valid" to "val" to standardize
    labels["val"] = labels.pop("valid")

    # Save labels and charset
    save_annotations(labels, charset, output_folder / output_dataset_name)

    # Remove temporary working directory
    shutil.rmtree(output_path / "temp")


def preformat_dataset(
    path: Path, rectify_orientation: bool = True, add_language_tokens: bool = True
) -> Dict:
    """
    Preformat the annotation of the whole dataset.
    :param path: Path for the formatted dataset.
    :param rectify_orientation: Whether to rotate the document image if originally inclined.
    :param add_language_tokens: Whether to include language tokens in annotation or not.
    :return: Dict with representation of whole documents for train/val/test.
    """
    dataset = {"train": list(), "valid": list(), "test": list()}

    language_tokens = {
        "french": "<language-french>",
        "english": "<language-english>",
        "arabic": "<language-arabic>",
        "other": "<language-unknown>",
    }

    for set_name in dataset.keys():
        (path / set_name).mkdir(exist_ok=True)
        xml_fold_path = path / "temp" / set_name / "XML" / "xml"
        # Read all XML files corresponding to given set (train, valid, test)
        for xml_path in tqdm(sorted(xml_fold_path.iterdir()), desc=f"Pre-formatting {set_name} set"):
            xml_root = ET.parse(xml_path).getroot()
            doc = xml_root.find(
                "{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_DOCUMENT"
            )
            document = dict()
            document["num_pages"] = int(doc.attrib["NrOfPages"])
            document["pages"] = list()
            document["languages"] = set()
            document["scripts"] = set()
            # Parse all pages for a given document and gather information such as languages and categories
            for id_page, page in enumerate(
                doc.findall("{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_PAGE")
            ):
                page_info = dict()
                page_info["width"] = int(page.attrib["width"])
                page_info["height"] = int(page.attrib["height"])
                page_info["languages"] = set()
                if id_page == 0:
                    document["category"] = page.attrib["Category"]
                page_info["category"] = document["category"]
                for i in range(5):
                    field_name = "Language_{}".format(i)
                    if field_name in page.attrib and page.attrib[field_name] != "":
                        page_info["languages"].add(page.attrib[field_name])
                document["languages"] = document["languages"].union(
                    page_info["languages"]
                )
                if len(page_info["languages"]) == 0:
                    page_info["languages"] = document["languages"]

                page_info["scripts"] = set()
                page_info["orientation"] = (
                    int(page.attrib["GEDI_orientation"])
                    if "GEDI_orientation" in page.attrib
                    else 0
                )
                img_name = page.attrib["src"]
                document["img_path"] = path / "temp" / set_name / "TIFF" / img_name

                # Parse all zones (text and graphics
                zones = page.findall(
                    "{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_ZONE"
                )
                text_regions = [
                    z for z in zones if z.attrib["gedi_type"] == "TextRegion"
                ]
                graphic_regions = [
                    z for z in zones if z.attrib["gedi_type"] == "GraphicRegion"
                ]
                page_info["text_regions"] = list()
                page_info["graphic_regions"] = list()

                # format text regions
                for text_region in text_regions:
                    # Rectify text orientation
                    if (
                        page_info["orientation"] != 0
                        and "orientationD" in text_region.attrib
                    ):
                        correct_text_region(text_region, page)
                    text_region_description = dict()
                    # ignore text region without text annotation
                    if (
                        "contents" not in text_region.attrib
                        or text_region.attrib["contents"] == ""
                    ):
                        continue
                    if "script" in text_region.attrib:
                        page_info["scripts"].add(text_region.attrib["script"])
                    text_region_description["label"] = convert_label(
                        text_region.attrib["contents"]
                    )
                    text_region_description["id"] = int(text_region.attrib["id"])
                    text_region_description["script"] = text_region.attrib["script"]
                    text_region_description["language"] = text_region.attrib["language"]

                    # Parse reading order annotation
                    if "nextZoneID" in text_region.attrib:
                        text_region_description["id_next"] = int(
                            text_region.attrib["nextZoneID"]
                        )
                    else:
                        text_region_description["id_next"] = None
                    rect_coord, poly_coord = region_to_coord(text_region)
                    if rectify_orientation and page_info["orientation"] != 0:
                        rect_coord = rectify_oriented_coord(
                            rect_coord,
                            page_info["orientation"],
                            page_info["width"],
                            page_info["height"],
                        )
                        if poly_coord:
                            poly_coord = rectify_oriented_coord(
                                poly_coord,
                                page_info["orientation"],
                                page_info["width"],
                                page_info["height"],
                            )
                    text_region_description["rect_coord"] = rect_coord
                    text_region_description["polygon_coord"] = poly_coord
                    text_region_description["functions"] = list()
                    for i in range(5):
                        field_name = "function_{}".format(i)
                        if (
                            field_name in text_region.attrib
                            and text_region.attrib[field_name] != ""
                        ):
                            text_region_description["functions"].append(
                                text_region.attrib[field_name]
                            )
                    page_info["text_regions"].append(text_region_description)

                # format graphic regions
                for graphic_region in graphic_regions:
                    graphic_region_description = dict()
                    coord = region_to_coord(graphic_region)
                    graphic_region_description["coord"] = coord
                    graphic_region_description["id"] = int(graphic_region.attrib["id"])
                    graphic_region_description["functions"] = list()
                    page_info["graphic_regions"].append(graphic_region_description)
                    for i in range(5):
                        field_name = "function_{}".format(i)
                        if (
                            field_name in graphic_region.attrib
                            and graphic_region.attrib[field_name] != ""
                        ):
                            graphic_region_description["functions"].append(
                                graphic_region.attrib[field_name]
                            )

                # Add type of next entity to read
                for i in range(len(page_info["text_regions"])):
                    text_ids = [tr["id"] for tr in page_info["text_regions"]]
                    graphic_ids = [gr["id"] for gr in page_info["graphic_regions"]]
                    if page_info["text_regions"][i]["id_next"]:
                        if page_info["text_regions"][i]["id_next"] in text_ids:
                            page_info["text_regions"][i]["id_next_type"] = "text"
                        elif page_info["text_regions"][i]["id_next"] in graphic_ids:
                            page_info["text_regions"][i]["id_next_type"] = "graphic"
                        else:
                            page_info["text_regions"][i]["id_next_type"] = "unknown"

                document["scripts"] = document["scripts"].union(page_info["scripts"])
                if len(page_info["scripts"]) == 0:
                    page_info["scripts"] = document["scripts"]
                document["pages"].append(page_info)
            if len(document["pages"]) != 0:
                # Order text regions to preserve reading order (based on text region positions
                document = order_text_regions(document)
                # Merge text region annotation to have page-level annotation
                for i in range(len(document["pages"])):
                    if len(document["pages"][i]["text_regions"]) == 0:
                        document["pages"][i]["label"] = list()
                        continue
                    last_language = document["pages"][i]["text_regions"][0]["language"]
                    page_label = (
                        [
                            language_tokens[last_language],
                        ]
                        if add_language_tokens
                        else list()
                    )
                    for tr in document["pages"][i]["text_regions"]:
                        if add_language_tokens and tr["language"] != last_language:
                            last_language = tr["language"]
                            page_label.append(language_tokens[last_language])
                        page_label.extend(
                            list(tr["label"]) + ["\n"]
                        )
                    del page_label[-1]
                    document["pages"][i]["label"] = page_label
                dataset[set_name].append(document)
    return dataset


def convert_label(label: str) -> str:
    """
    Remove/correct some special characters from annotation.
    :params label: The original label.
    :return: The corrected label.
    """
    label = re.sub("¤{([^¤]*)[/|]([^¤]*)}¤", r"\1", label, flags=re.DOTALL)
    label = re.sub("¤([^¤]*)¤", r"\1", label, flags=re.DOTALL)
    sub = {
        "\xa0": "",
        "…": "...",
        "—": "-",
        "№": "N°",
        "\t": "",
        "▶": "",
        "►": "",
        "●": "",
        "": "",
    }
    for key in sub:
        label = label.replace(key, sub[key])
    return label


def rectify_oriented_coord(
    coord: dict, orientation: int, page_width: int, page_height: int
) -> dict:
    """
    Rectify coordinates to match rectified document orientation.
    :param coord: original coordinates.
    :param orientation: The degree of orientation.
    :param page_width: The width of the page.
    :param page_height: The height of the page.
    :return: The updated coordinates that match the re-oriented document image.
    """
    if coord["type"] == "polygon":
        new_points = list()
        for point in coord["points"]:
            new_points.append(
                rectify_oriented_point(point, orientation, page_width, page_height)
            )
        coord["points"] = np.array(new_points, dtype=np.int64)
        return coord
    else:
        if orientation == 180:
            coord["left"], coord["right"] = (
                page_width - coord["right"],
                page_width - coord["left"],
            )
            coord["bottom"], coord["top"] = (
                page_height - coord["top"],
                page_height - coord["bottom"],
            )
            return coord
        elif orientation == 90:
            coord["left"], coord["right"], coord["top"], coord["bottom"] = (
                page_height - coord["bottom"],
                page_height - coord["top"],
                coord["left"],
                coord["right"],
            )
            return coord
        elif orientation == 270:
            coord["left"], coord["right"], coord["top"], coord["bottom"] = (
                coord["top"],
                coord["bottom"],
                page_width - coord["right"],
                page_width - coord["left"],
            )
            return coord


def rectify_oriented_point(
    point: np.ndarray, orientation: int, width: int, height: int
) -> list:
    """
    Rectify 2D point to match rectified document orientation.
    :param point: Numpy array with two integers (x, y).
    :param orientation: The angle of orientation.
    :param width: The page width.
    :param height: The page height.
    :return: A list of two integers [x, y].
    """
    if orientation == 180:
        return [width - point[0], height - point[1]]
    elif orientation == 90:
        return [height - point[1], point[0]]
    elif orientation == 270:
        return [point[1], width - point[0]]


def correct_text_region(
    text_region: xml.etree.ElementTree.Element, page: xml.etree.ElementTree.Element
) -> xml.etree.ElementTree.Element:
    """
    Rectify region orientation.
    :param text_region: XML entity for text region.
    :param page: XML entity for page.
    :return: Text region XML entity with corrected position.
    """
    angle = int(page.attrib["GEDI_orientation"])
    if angle == 180:
        text_region.attrib["col"] = str(
            int(text_region.attrib["col"]) - int(text_region.attrib["width"])
        )
        text_region.attrib["row"] = str(
            int(text_region.attrib["row"]) - int(text_region.attrib["height"])
        )
    elif angle == 90:
        text_region.attrib["col"] = str(
            int(text_region.attrib["row"]) - int(text_region.attrib["width"])
        )
        text_region.attrib["row"] = str(
            int(page.attrib["width"]) - int(text_region.attrib["col"])
        )
    return text_region


def region_to_coord(region: xml.etree.ElementTree.Element) -> dict:
    """
    Standardize region coordinates (whether it is rectangle or polygon).
    :param region: XML entity representing a region.
    :return: Coordinate dict.
    """
    if all(a in region.attrib for a in ["col", "row", "width", "height"]):
        rect_coord = text_region_to_coord(region)
        poly_coord = None
    else:
        rect_coord = polygon_to_rect_coord(region)
        poly_coord = polygon_to_coord(region)
    return rect_coord, poly_coord


def polygon_to_coord(region: np.ndarray) -> dict:
    """
    Standardize polygon coordinates.
    :param region: Array of 2D points that represent a polygon.
    :return: Coordinates dict for polygon.
    """
    polygon = [
        point.replace("(", "").replace(")", "")
        for point in region.attrib["polygon"].split(";")
    ]
    polygon = [np.array([int(a.split(",")[0]), int(a.split(",")[1])]) for a in polygon]
    polygon = np.stack(polygon)
    coord = {
        "points": polygon,
        "orientation": -float(region.attrib["orientationD"])
        if "orientationD" in region.attrib
        else 0,
        "type": "polygon",
    }
    return coord


def polygon_to_rect_coord(region: xml.etree.ElementTree.Element) -> dict:
    """
    Extract text region coordinates from polygon region XML entity.
    :param region: XML polygon region entity.
    :return: Coordinates dict of region as top, bottom, right and left.
    """
    polygon = [
        a.replace("(", "").replace(")", "") for a in region.attrib["polygon"].split(";")
    ]
    polygon = [(int(a.split(",")[0]), int(a.split(",")[1])) for a in polygon]

    left = min([int(a[0]) for a in polygon])
    right = max([int(a[0]) for a in polygon])
    top = min([int(a[1]) for a in polygon])
    bottom = max([int(a[1]) for a in polygon])
    coord = {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "orientation": 0,
        "type": "rect",
    }
    return coord


def text_region_to_coord(region: xml.etree.ElementTree.Element) -> dict:
    """
    Extract text region coordinates from text region XML entity.
    :param region: XML text region entity.
    :return: Coordinates dict of region as top, bottom, right and left.
    """
    col, row, width, height = [
        int(region.attrib[a]) for a in ["col", "row", "width", "height"]
    ]
    coord = {
        "left": col,
        "right": col + width,
        "top": row,
        "bottom": row + height,
        "orientation": -float(region.attrib["orientationD"])
        if "orientationD" in region.attrib
        else 0,
        "type": "rect",
    }
    return coord


def filter_dataset(
    dataset: dict,
    languages: list,
    categories: list,
    scripts: list,
    a4_only: bool = True,
    filter_orientation: bool = True,
) -> dict:
    """
    Select only a subset of the MAURDOR dataset based on languages, categories and scripts.
    :param languages: List of languages to keep.
    :param categories: List of categories to keep.
    :param scripts: List of scripts to keep.
    :param a4_only: Whether to only keep page that follows A format or not.
    :param filter_orientation: Whether to only keep page that are not inclined or not.
    :return: The filtered pre-formatted dataset.
    """
    new_dataset = dataset.copy()
    for set_name in dataset.keys():
        new_set = list()
        for i, sample in enumerate(dataset[set_name]):
            to_del = False
            if languages is not None and any(
                [lang not in languages for lang in sample["languages"]]
            ):
                continue
            if categories is not None and sample["category"] not in categories:
                continue
            if scripts is not None and any(
                [s not in scripts for s in sample["scripts"]]
            ):
                continue
            for page_id, page in enumerate(sample["pages"]):
                # Remove documents that are surely not in A4 format
                if a4_only:
                    h, w = page["height"], page["width"]
                    ratio = max([h, w]) / min([h, w])
                    # A4 format = 29.7/21 = 1,41 (height/width ratio)
                    if not 1.3 < ratio < 1.5:
                        to_del = True
                        break

                    # Remove documents which contain text region with wrong orientation
                    if filter_orientation:
                        for text_region in page["text_regions"]:
                            if (
                                not text_region["polygon_coord"]
                                and text_region["rect_coord"]["orientation"] != 0
                            ):
                                to_del = True
                                break

            if not to_del:
                new_set.append(dataset[set_name][i])
        new_dataset[set_name] = new_set
    return new_dataset


def order_text_regions(sample: dict) -> dict:
    """
    Order text regions based on coordinates.
    :param sample: A dict representation of a document.
    :return: The updated document representation for which text regions have been ordered for coherent reading order.
    """

    def gather_by_id(gr, list_id, list_gr):
        new_list = list()
        gr_to_gather = list()
        for i, g in enumerate(list_gr):
            if i in list_id:
                gr_to_gather.append(g)
            else:
                new_list.append(g)
        new_list.append(
            {
                "coord": gather_tr_coords(
                    gr_to_gather
                    + [
                        gr,
                    ]
                ),
                "atomic_items": [
                    gr,
                ],
            }
        )
        for tr in gr_to_gather:
            new_list[-1]["atomic_items"].extend(tr["atomic_items"])
        return new_list

    def merge_groups_as_rows(list_group):
        def is_same_row(gr1, gr2):
            higher, lower = (
                (gr1["coord"], gr2["coord"])
                if gr1["coord"]["top"] < gr2["coord"]["top"]
                else (gr2["coord"], gr1["coord"])
            )
            h_high = higher["bottom"] - higher["top"]
            h_low = lower["bottom"] - lower["top"]
            return min([h_high, h_low, higher["bottom"] - lower["top"]]) > 0.25 * min(
                [h_high, h_low]
            )

        new_rows = list()
        for gr in list_group:
            if len(new_rows) == 0:
                new_rows.append(
                    {
                        "coord": gr["coord"],
                        "atomic_items": [
                            gr,
                        ],
                    }
                )
            else:
                group_with = list()
                for id_gr, ngr in enumerate(new_rows):
                    if is_same_row(gr, ngr):
                        group_with.append(id_gr)

                if len(group_with) == 0:
                    new_rows.append(
                        {
                            "coord": gr["coord"],
                            "atomic_items": [
                                gr,
                            ],
                        }
                    )

                else:
                    new_rows = gather_by_id(gr, group_with, new_rows)
        return new_rows

    def gather_form_key_values(text_regions):
        tr_groups = list()
        used_ids = list()
        next_ids = [t["id_next"] for t in text_regions if t["id_next"]]
        for tr in text_regions:
            if tr["id"] in used_ids or tr["id"] in next_ids:
                continue
            used_ids.append(tr["id"])
            if tr["id_next"] and tr["id_next_type"] == "text":
                list_consecutive_items = [
                    tr,
                ]
                temp_tr = tr
                while temp_tr["id_next"] and temp_tr["id_next_type"] == "text":
                    used_ids.append(temp_tr["id_next"])
                    temp_tr = [
                        t for t in text_regions if t["id"] == temp_tr["id_next"]
                    ][0]

                    list_consecutive_items.append(temp_tr)

                tr_groups.append(
                    {
                        "coord": gather_tr_coords(list_consecutive_items),
                        "regions": list_consecutive_items,
                    }
                )
            else:
                tr_groups.append(
                    {
                        "coord": tr["rect_coord"],
                        "regions": [
                            tr,
                        ],
                    }
                )
        return tr_groups

    def split_in_cols(row):
        def is_same_col(gr1, gr2):
            lefter, righter = (
                (gr1["coord"], gr2["coord"])
                if gr1["coord"]["left"] < gr2["coord"]["left"]
                else (gr2["coord"], gr1["coord"])
            )
            l_width = lefter["right"] - lefter["left"]
            r_width = righter["right"] - righter["left"]
            return min(
                [l_width, r_width, lefter["right"] - righter["left"]]
            ) > 0.25 * min([r_width, l_width])

        list_cols = list()
        for item in row["atomic_items"]:
            if len(list_cols) == 0:
                list_cols.append(
                    {
                        "coord": item["coord"],
                        "atomic_items": [
                            item,
                        ],
                    }
                )
            else:
                group_with = list()
                for id_gr, ngr in enumerate(list_cols):
                    if is_same_col(item, ngr):
                        group_with.append(id_gr)
                if len(group_with) == 0:
                    list_cols.append(
                        {
                            "coord": item["coord"],
                            "atomic_items": [
                                item,
                            ],
                        }
                    )
                else:
                    list_cols = gather_by_id(item, group_with, list_cols)
        row["cols"] = list_cols
        del row["atomic_items"]
        return row

    def gather_tr_coords(list_tr):
        return {
            "top": min(
                [
                    g["coord"]["top"] if "coord" in g else g["rect_coord"]["top"]
                    for g in list_tr
                ]
            ),
            "bottom": max(
                [
                    g["coord"]["bottom"] if "coord" in g else g["rect_coord"]["bottom"]
                    for g in list_tr
                ]
            ),
            "left": min(
                [
                    g["coord"]["left"] if "coord" in g else g["rect_coord"]["left"]
                    for g in list_tr
                ]
            ),
            "right": max(
                [
                    g["coord"]["right"] if "coord" in g else g["rect_coord"]["right"]
                    for g in list_tr
                ]
            ),
        }

    new_sample = sample.copy()
    new_pages = list()
    for page in sample["pages"]:
        new_page = page.copy()
        # group form key/value together and encapsulate each tr into group tr
        atomic_items = gather_form_key_values(page["text_regions"])
        rows = merge_groups_as_rows(atomic_items)
        split_rows = [split_in_cols(r) for r in rows]
        ordered_text_regions = list()
        split_rows.sort(key=lambda x: x["coord"]["top"])
        for row in split_rows:
            row["cols"].sort(key=lambda x: x["coord"]["left"])
            for col in row["cols"]:
                col["atomic_items"].sort(key=lambda x: x["coord"]["top"])
                for item in col["atomic_items"]:
                    for tr in item["regions"]:
                        ordered_text_regions.append(tr)
        new_page["text_regions"] = ordered_text_regions
        new_pages.append(new_page)
    new_sample["pages"] = new_pages
    return new_sample


def change_a4_page_img_resolution(page: dict, img: np.ndarray, dpi: int = 150) -> tuple:
    """
    Change resolution to given dpi by assuming an A4 input format (11.7 x 8.3 inches).
    :param page: Dict representing a page.
    :param img: The document image.
    :param dpi: The image output resolution.
    :return: Tuple (page, image) where page is the updated page representation and image is the resized image.
    """
    h, w = img.shape[:2]
    ratio_h, ratio_w = (
        (11.7 * dpi / h, 8.3 * dpi / w) if h > w else (8.3 * dpi / h, 11.7 * dpi / w)
    )
    new_h, new_w = int(ratio_h * h), int(ratio_w * w)
    img = Image.fromarray(img)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(img)
    page["width"] = new_w
    page["height"] = new_h
    for tr in page["text_regions"]:
        for coord in ["top", "bottom"]:
            tr["rect_coord"][coord] = int(ratio_h * tr["rect_coord"][coord])
        for coord in ["left", "right"]:
            tr["rect_coord"][coord] = int(ratio_w * tr["rect_coord"][coord])
        if tr["polygon_coord"]:
            tr["polygon_coord"]["points"] = (
                tr["polygon_coord"]["points"] * np.array([ratio_w, ratio_h])
            ).astype(np.int64)
    return page, img


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate an image of a given angle.
    :param img: The input image.
    :param angle: The angle of rotation.
    :return: The rotated image.
    """
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
