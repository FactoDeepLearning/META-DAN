"""
Module dedicated to generate synthetic documents following a rather generic layout, for MAURDOR C3 / C4 datasets.
"""

from dar.generic.utils import randint, resize_max, pad_image_width_random, rand, randint_normal
from scipy.ndimage import rotate
from dar.text_recognition.synthetic.synthetic_line import generate_typed_text_line_image_default_config, DEFAULT_SYN_CONFIG
import json
import numpy as np


class MAURDORSynthetic:

    def __init__(self, font_manager, label_path):
        self.font_manager = font_manager
        self.labels = self.load_labels(label_path)
        self.dataset_name = None
        self.padding_value = DEFAULT_SYN_CONFIG["background_color_default"][0]

    def load_labels(self, label_path):
        res = list()
        with open(label_path) as f:
            for sample in json.load(f)["train"].values():
                res.append("".join(sample["text"]))
        return res

    def render(self, sample, nb_lines, crop=False, text_lines=None):
        self.dataset_name = sample["dataset"]
        img_height, img_width, c = sample["img"].shape
        background = np.ones((img_height, img_width, c), dtype=sample["img"].dtype) * 255
        page_label = list()
        min_top_crop = img_height
        max_bottom_crop = 0
        min_line_height = 30
        orientation = "landscape" if img_height < img_width else "portrait"
        if orientation == "portrait":
            max_len_line = 100
        else:
            max_len_line = 150
        last_pg = {
            "top": 0,
            "bottom": 0,
            "right": 0,
            "height": 0,
        }
        while nb_lines > 0:
            if img_width - last_pg["right"] > 0.25 * img_width:
                block_mode = np.random.choice(["bottom", "right_side"], 1, p=[0.4, 0.6])[0]
            else:
                block_mode = "bottom"
            if block_mode == "bottom":
                min_top = last_pg["bottom"]
                min_left = 0
                max_left = int(0.6 * img_width)
            else:
                min_top = min(img_height, last_pg["top"] + randint(20, 30))
                min_left = int(last_pg["right"] + 0.02 * img_width)
                max_left = int(0.8 * img_width)

            num_line_block = randint(1, nb_lines + 1)
            num_line_block = min(num_line_block, (img_height - min_top) // min_line_height)
            if num_line_block == 0:
                break
            left = randint(min_left, max_left + 1)
            max_len_line_block = max(1, int((img_width - left) / img_width * max_len_line))
            line_labels = self.get_lines_maurdor(num_line_block, max_len_line=max_len_line_block)

            nb_lines -= num_line_block
            max_width = img_width - left
            max_top = max(min(img_height, img_height - min_line_height * num_line_block), min_top)
            if max_top < 0:
                continue

            top = randint(min_top, max_top + 1)

            pg_img = self.generate_typed_text_paragraph_image(line_labels, max_pad_left_ratio=0.1, same_font_size=True)
            if rand() < 0.2:
                rotation_degree = randint_normal(0, 10, -10, 10)
                pg_img = rotate(pg_img, angle=rotation_degree, reshape=True, cval=self.padding_value)
            pg_img = resize_max(pg_img, max_height=int(img_height - top), max_width=int(max_width))
            h, w = pg_img.shape[:2]
            bottom = top + h
            right = left + w
            last_pg = {
                "top": top,
                "bottom": bottom,
                "right": right,
                "height": h,
            }
            background[top:bottom, left:right, ...] = pg_img
            page_label.extend(line_labels)
            min_top_crop = min(min_top_crop, top)
            max_bottom_crop = max(max_bottom_crop, bottom)

        if crop and min_top_crop < max_bottom_crop:
            background = background[min_top_crop:max_bottom_crop]

        page_label = "\n".join(page_label)

        return background, page_label

    def get_lines_maurdor(self, num_lines, min_len_line=1, max_len_line=100):
        line_labels = list()
        while len(line_labels) != num_lines:
            line_label = ""
            while len(line_label) < min_len_line or len(line_label) > max_len_line:
                line_label = []
                while np.sum([len(l) for l in line_label]) < min_len_line:
                    sample = self.labels[randint(0, len(self.labels))]
                    lines = sample.split("\n")
                    line_label.append(lines[randint(0, len(lines))])
                line_label = " ".join(line_label)
                if len(line_label) > max_len_line:
                    line_label = " ".join(line_label[:max_len_line].split(" ")[:-1])
            line_labels.append(line_label)
        return line_labels

    def generate_typed_text_paragraph_image(self, texts, max_pad_left_ratio=0.1, same_font_size=False):
        if same_font_size:
            images = list()
            font_size = randint(DEFAULT_SYN_CONFIG["font_size_min"], DEFAULT_SYN_CONFIG["font_size_max"] + 1)
            for text in texts:
                img = generate_typed_text_line_image_default_config(text, font_path=self.font_manager.get_random_font(text), font_size=font_size)
                images.append(img)
        else:
            images = [generate_typed_text_line_image_default_config(t, font_path=self.font_manager.get_random_font(t)) for t in texts]

        max_width = max([img.shape[1] for img in images])

        padded_images = [pad_image_width_random(img, max_width, padding_value=self.padding_value, max_pad_left_ratio=max_pad_left_ratio) for img in images]
        return np.concatenate(padded_images, axis=0)
