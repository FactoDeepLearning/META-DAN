# -*- coding: utf-8 -*-
"""
Utils function.
"""

import re
from pathlib import Path
from typing import Any, Dict

from lxml import etree
from PIL import Image
import cv2
import numpy as np
import json
import pickle
import os


def _parse_namespaces(root_node: etree.Element) -> Dict[Any, str]:
    """
    Read the namespaces map in the parsed file. There should only be one and it will be used as default prefix.
    :param root_node: Root node of the parsed XML file.
    :return: The namespace map that should be used when querying the XML.
    """
    namespaces = set(root_node.nsmap.values())
    return {None: namespaces.pop()}


def clean_text_label(label: str) -> str:
    """
    Remove extra space or line break characters.
    :param label: Text annotation.
    :return: The text annotation from which extra space / line breaks are removed.
    """
    label = html_replace(label)
    temp = re.sub("\n[\s\n]*\n", "\n", label) # remove duplicated new line chars
    temp = temp.replace('­', "")
    temp = re.sub("( )+", " ", temp).strip(" \n") # remove duplicated spaces
    return "\n".join([l.strip() for l in temp.split("\n")]) # trim each line (in case of multi-line label)



def load_resize_dpi_save_image(
    source_path: Path, target_path: Path, source_dpi: int, target_dpi: int
):
    """
    Load image, apply resolution modification and save it.
    :param source_path: Path of the input image.
    :param target_path: Path of the output image, after resizing.
    :param source_dpi: Resolution of the input image.
    :param target_dpi: Expected resolution as output.
    """
    img = Image.open(source_path)
    img = resize_image_dpi(img, source_dpi, target_dpi)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(target_path)


def resize_image_dpi(img: Image, source_dpi: int, target_dpi: int) -> Image:
    """
    Apply resolution modification to image.
    :param img: The original image.
    :param source_dpi: The resolution of the original image.
    :param target_dpi: The expected resolution as output.
    """
    if source_dpi != target_dpi:
        ratio = target_dpi / source_dpi
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)), Image.BILINEAR
        )
    return img


def natural_sort(str_list: list[str]) -> list[str]:
    """
    Sort list of string by considering integer order.
    Example: ["test_1", "test_11", "test_2"] becomes ["test_1", "test_2", "test_11"].
    :param str_list: List of string.
    :return: The sorted list.
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return sorted(
        str_list, key=lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    )


def html_replace(text):
    mapping = {
        "&amp;": "&",
        "&quot;": "\"",
        "&apos;": "'",
        "&gt;": ">",
        "&lt;": "<"
    }

    for key in mapping.keys():
        text = text.replace(key, mapping[key])
    return text


def crop_polygon(source_image: np.ndarray, points: list[tuple[int, int]]):
    img = source_image.copy()
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(points).astype(np.int32)
    cv2.fillPoly(mask, [points, ], (255))
    mask = mask.astype(bool)
    img[~mask] = 255
    rect = cv2.boundingRect(points)
    cropped = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def coord_str_to_points(coord_str: str) -> list[tuple[int, int]]:
    """
    Extract bounding box from string representing coordinates.
    :param coord_str: String representing coordinates.
    """
    points = coord_str.split(" ")
    x_points, y_points = zip(
        *[[int(point.split(",")[0]), int(point.split(",")[1])] for point in points]
    )
    return [(x, y) for x, y in zip(x_points, y_points)]


def coord_str_to_bounding_box(coord_str: str) -> Dict[str, int]:
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


# def is_recognizable_with_tesseract(line_img, str_y, cer_threshold=0.5):
#     return compute_cer_tesseract(line_img, str_y) <= cer_threshold


def compute_cer_tesseract(line_img, str_y, language):
    import pytesseract
    import editdistance
    str_x = pytesseract.image_to_string(line_img, lang=language)
    cer = editdistance.eval(str_x, str_y)/len(str_y)
    return cer


def rotate_best_cer(line_img, str_y, language):
    right_rotate = cv2.rotate(line_img.copy(), cv2.ROTATE_90_CLOCKWISE)
    left_rotate = cv2.rotate(line_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cer_list = [compute_cer_tesseract(img, str_y, language) for img in [line_img, right_rotate, left_rotate]]
    print(cer_list)
    min_cer = np.min(cer_list)
    if cer_list[0] == min_cer:
        return line_img
    if cer_list[1] == min_cer:
        return right_rotate
    return left_rotate


def save_annotations(labels, charset, dataset_path):
    (dataset_path / "labels.json").write_text(
        json.dumps(labels, sort_keys=True, indent=4)
    )
    (dataset_path / "charset.pkl").write_bytes(
        pickle.dumps(sorted(list(charset)))
    )


def extract_page_transcription(
    xml_page_anns: etree.Element, namespaces: Dict[Any, str]
) -> dict:
    """
    Extract and concatenate the transcriptions of all the lines on a page.
    Lines are separated by a newline character "\n".
    :param xml_page_anns: Root node of the annotation XML file.
    :param namespaces: The namespace map used to query the XML file.
    :return: The full page transcription.
    """
    # Concatenate all lines transcriptions
    page = xml_page_anns.find(".//Page", namespaces=namespaces)
    page_dict = {
        "text": None,
        "text_regions": []
    }
    for text_region in page.findall(".//TextRegion", namespaces=namespaces):
        region_dict = {
            "text": None,
            "lines": []
        }
        for text_line in text_region.findall(".//TextLine", namespaces=namespaces):
            line_dict = {
                "text": None,
                "words": []
            }
            for word in text_line.findall(".//Word", namespaces=namespaces):
                word_label = word.find(".//Unicode", namespaces=namespaces).text
                if word_label is None:
                    continue
                word_coords = word.find(".//Coords", namespaces=namespaces).attrib["points"]
                word_points = [ (int(point_str.split(",")[0]), int(point_str.split(",")[1])) for point_str in word_coords.split(" ")]
                word_dict = {
                    "text": word_label,
                    "coords": {
                        "bottom": max([p[1] for p in word_points]),
                        "top": min([p[1] for p in word_points]),
                        "right": max([p[0] for p in word_points]),
                        "left": min([p[0] for p in word_points]),
                    }
                }
                line_dict["words"].append(word_dict)
            line_dict["text"] = " ".join([w["text"] for w in line_dict["words"]])
            line_dict["coords"] = {
                "bottom": max([w["coords"]["bottom"] for w in line_dict["words"]]),
                "top": min([w["coords"]["top"] for w in line_dict["words"]]),
                "right": max([w["coords"]["right"] for w in line_dict["words"]]),
                "left": min([w["coords"]["left"] for w in line_dict["words"]]),
            }
            region_dict["lines"].append(line_dict)
        region_dict["text"] = "\n".join([l["text"] for l in region_dict["lines"]])
        region_dict["coords"] = {
            "bottom": max([l["coords"]["bottom"] for l in region_dict["lines"]]),
            "top": min([l["coords"]["top"] for l in region_dict["lines"]]),
            "right": max([l["coords"]["right"] for l in region_dict["lines"]]),
            "left": min([l["coords"]["left"] for l in region_dict["lines"]]),
        }
        page_dict["text_regions"].append(region_dict)
    page_dict["text"] = "\n".join([tr["text"] for tr in page_dict["text_regions"]])
    page_dict["coords"] = {
        "bottom": max([tr["coords"]["bottom"] for tr in page_dict["text_regions"]]),
        "top": min([tr["coords"]["top"] for tr in page_dict["text_regions"]]),
        "right": max([tr["coords"]["right"] for tr in page_dict["text_regions"]]),
        "left": min([tr["coords"]["left"] for tr in page_dict["text_regions"]]),
    }
    return page_dict

def export_split_from_annotation(file_path: str, output_path: str) -> None:
    """
    Extract file names from annotation file and export it as a json file
    {"train": list of file names, "val": list of file names, "test": list of file names}
    """
    with open(file_path, "rb") as fin:
        gt = json.load(fin)
    splits = {}
    for set_name in ["train", "val", "test"]:
        splits[set_name] = [os.path.basename(path) for path in gt[set_name].keys()]
    with open(output_path, "w") as fout:
        json.dump(splits, fout, sort_keys=True, indent=4)