"""
Module dedicated to generate synthetic document following the layout of RIMES 2009/2024 datasets.
"""

from dar.dataset_format.rimes2009 import SEM_MATCHING_TOKENS, SEM_START_TOKENS, order_text_regions
from dar.generic.utils import randint, rand_uniform, resize_max, pad_image_width_random
from dar.text_recognition.synthetic.synthetic_line import generate_typed_text_line_image_default_config, DEFAULT_SYN_CONFIG
from dar.text_recognition.synthetic.synthetic_doc import concatenate_lines_overlap
from dar.generic.transforms import apply_transform, Tightening, RandomRotation
import random
import re
import json
import numpy as np


class RIMESSynthetic:

    text_types = list(SEM_START_TOKENS.keys())
    missing_upper_chars = {
        "È": "E",
        "Ë": "E",
        "Û": "U",
        "Ù": "U",
        "Ü": "U",
        "Î": "I",
        "Ï": "I",
        "Â": "A",
        "Œ": "OE",
        "Ö": "O",
        "Ø": "ø",
    }

    def __init__(self, font_manager, sem_label_path, add_sem_tokens=False, enable_overlap=False):
        self.font_manager = font_manager
        self.add_sem_tokens = add_sem_tokens
        self.layout_tokens = "".join(list(SEM_MATCHING_TOKENS.values()) + list(SEM_START_TOKENS.values()))
        self.labels, self.num_samples = self.load_labels(sem_label_path)
        self.sem_stats = self.compute_sem_stat()
        self.dataset_name = None
        self.overlap = enable_overlap

    def load_labels(self, label_path):
        labels_by_type = dict()
        patterns_by_type = dict()
        for text_type in self.text_types:
            labels_by_type[text_type] = list()
            t_start = SEM_START_TOKENS[text_type]
            t_end = SEM_MATCHING_TOKENS[SEM_START_TOKENS[text_type]]
            patterns_by_type[text_type] = re.compile(f"{t_start}([^{t_start}{t_end}]*){t_end}", flags=re.DOTALL)

        with open(label_path, "rb") as f:
            labels = json.load(f)
        num_samples = len(labels["train"])
        for sample in labels["train"].values():
            text = "".join(sample["text"])
            for text_type in self.text_types:
                new_text_list = patterns_by_type[text_type].findall(text)
                new_text_list = [re.sub(f"[{self.layout_tokens}]", "", t) for t in new_text_list]
                new_text_list = [t.replace("\n \n", "\n") for t in new_text_list]
                labels_by_type[text_type].extend(new_text_list)
        labels_by_type["Reference"] = [t for t in labels_by_type["Reference"] if len(t) < 50 and "\n" not in t]
        labels_by_type["Coordonnées Expéditeur"] = [t for t in labels_by_type["Coordonnées Expéditeur"] if not(len(t) < 50 and "\n" not in t)]
        return labels_by_type, num_samples

    def render(self, sample, nb_lines=20, crop=False, max_len=100, text_lines=None):
        self.dataset_name = sample["dataset"]
        h, w, c = sample["img"].shape
        background = np.ones((h, w, c), dtype=sample["img"].dtype) * 255

        def larger_lines(label):
            lines = label.split("\n")
            new_lines = list()
            while len(lines) > 0:
                if len(lines) == 1:
                    new_lines.append(lines[0])
                    del lines[0]
                elif len(lines[0]) + len(lines[1]) < max_len:
                    new_lines.append("{} {}".format(lines[0], lines[1]))
                    del lines[1]
                    del lines[0]
                else:
                    new_lines.append(lines[0])
                    del lines[0]
            return "\n".join(new_lines)

        h, w, c = background.shape

        ordered_modes = ['Corps de texte', 'PS/PJ', 'Ouverture', 'Date, Lieu', 'Coordonnées Expéditeur',
                         'Coordonnées Destinataire', ]
        object_ref = ['Objet', 'Reference']
        random.shuffle(object_ref)
        ordered_modes = ordered_modes[:3] + object_ref + ordered_modes[3:]
        kept_modes = list()
        for mode in ordered_modes:
            if rand_uniform(0, 1) < self.sem_stats[mode]:
                kept_modes.append(mode)

        # generate paragraph labels
        paragraphs = dict()
        for mode in kept_modes:
            paragraphs[mode] = self.generate_paragraph(mode=mode, mix=True, text_lines=text_lines, num_all_lines=nb_lines)
            if mode == "Corps de texte" and rand_uniform(0, 1) < 0.4:
                for _ in range(2):
                    concat_line = randint(0, 2) == 0
                    if concat_line:
                        paragraphs[mode]["label"] = larger_lines(paragraphs[mode]["label"])
                while (len(paragraphs[mode]["label"].split("\n")) <= 30):
                    body2 = self.generate_paragraph(mode=mode, mix=True, text_lines=text_lines, num_all_lines=nb_lines)
                    paragraphs[mode]["label"] += "\n" + (larger_lines(body2["label"]) if concat_line else body2["label"])
                    paragraphs[mode]["label"] = "\n".join(paragraphs[mode]["label"].split("\n")[:40])
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            label_upper = paragraphs["Corps de texte"]["label"].upper()
            for key, val in self.missing_upper_chars.items():
                label_upper = label_upper.replace(key, val)
            paragraphs["Corps de texte"]["label"] = label_upper
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            labels = paragraphs["Corps de texte"]["label"].split("\n")
            duplicated_label = labels[randint(0, len(labels))]
            labels.insert(randint(0, len(labels)), duplicated_label)
            paragraphs["Corps de texte"]["label"] = "\n".join(labels)
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            paragraphs["Corps de texte"]["label"] = larger_lines(paragraphs["Corps de texte"]["label"])
        for mode in paragraphs.keys():
            line_labels = paragraphs[mode]["label"].split("\n")
            if len(line_labels) == 0:
                print("ERROR")
            paragraphs[mode]["lines"] = list()
            for line_label in line_labels:
                if len(line_label) > 100:
                    for chunk in [line_label[i:i + max_len] for i in range(0, len(line_label), max_len)]:
                        paragraphs[mode]["lines"].append(chunk)
                else:
                    paragraphs[mode]["lines"].append(line_label)
        page_labels = {
            "raw": [],
            "begin": [],
            "sem": []
        }

        top_limit = 0
        bottom_limit = h
        max_bottom_crop = 0
        min_top_crop = h
        has_opening = has_object = has_reference = False
        top_opening = top_object = top_reference = 0
        right_opening = right_object = right_reference = 0
        has_reference = False
        date_on_top = False
        date_alone = False

        # create paragraph images and place them on the page
        for mode in kept_modes:
            if nb_lines <= 0:
                del paragraphs[mode]
                continue
            pg = paragraphs[mode]
            if len(pg["lines"]) > nb_lines:
                pg["lines"] = pg["lines"][:nb_lines]
            nb_lines -= len(pg["lines"])
            pg_image = self.generate_typed_text_paragraph_image(pg["lines"], max_pad_left_ratio=1, same_font_size=True)
            if rand_uniform(0, 1) < 0.1:
                pg_image = apply_transform(pg_image, Tightening(color=255, remove_proba=0.75))
            if rand_uniform(0, 1) < 0.1:
                try:
                    pg_image = apply_transform(pg_image, RandomRotation(degrees=10, expand=True, fill=255))
                except:
                    print(f"Error labels: {pg['lines']}")
                    print(f"Mode: {mode}")
            pg["added"] = True
            if mode == 'Corps de texte':
                pg_image = resize_max(pg_image, max_height=int(0.5 * h), max_width=w)
                img_h, img_w = pg_image.shape[:2]
                min_top = int(0.4 * h)
                max_top = int(0.9 * h - img_h)
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_body = top + img_h
                top_body = top
                bottom_limit = min(top, bottom_limit)
            elif mode == "PS/PJ":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = bottom_body
                max_top = int(min(h - img_h, bottom_body + 0.15 * h))
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_limit = min(top, bottom_limit)
            elif mode == "Ouverture":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = int(top_body - 0.05 * h)
                max_top = top_body - img_h
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_opening = True
                top_opening = top
                right_opening = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Objet":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_reference - img_h if has_reference else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_object = True
                top_object = top
                right_object = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Reference":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_object - img_h if has_object else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15 * w), int(w - img_w)) + 1)
                has_reference = True
                top_reference = top
                right_reference = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == 'Date, Lieu':
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                if h - max_bottom_crop - 10 > img_h and randint(0, 10) == 0:
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w - img_w)
                else:
                    min_top = top_body - img_h
                    max_top = top_body - img_h
                    min_left = 0
                    # Check if there is anough place to put the date at the right side of opening, reference or object
                    if object_ref == ['Objet', 'Reference']:
                        have = [has_opening, has_object, has_reference]
                        rights = [right_opening, right_object, right_reference]
                        tops = [top_opening, top_object, top_reference]
                    else:
                        have = [has_opening, has_reference, has_object]
                        rights = [right_opening, right_reference, right_object]
                        tops = [top_opening, top_reference, top_object]
                    for right_r, top_r, has_r in zip(rights, tops, have):
                        if has_r:
                            if right_r + img_w >= 0.95 * w:
                                max_top = min(top_r - img_h, max_top)
                                min_left = 0
                            else:
                                min_left = max(min_left, right_r + 0.05 * w)
                                min_top = top_r - img_h if min_top == top_body - img_h else min_top
                    if min_left != 0 and randint(0, 5) == 0:
                        min_left = 0
                        for right_r, top_r, has_r in zip(rights, tops, have):
                            if has_r:
                                max_top = min(max_top, top_r - img_h)

                    max_left = max(min_left, w - img_w)

                    # No placement found at right-side of opening, reference or object
                    if min_left == 0:
                        # place on the top
                        if randint(0, 2) == 0:
                            min_top = 0
                            max_top = int(min(0.05 * h, max_top))
                            date_on_top = True
                        # place just before object/reference/opening
                        else:
                            min_top = int(max(0, max_top - 0.05 * h))
                            date_alone = True
                            max_left = min(max_left, int(0.1 * w))

                    min_top = min(min_top, max_top)
                    top = randint(min_top, max_top + 1)
                    left = randint(int(min_left), max_left + 1)
                    if date_on_top:
                        top_limit = max(top_limit, top + img_h)
                    else:
                        bottom_limit = min(top, bottom_limit)
                    date_right = left + img_w
                    date_bottom = top + img_h
            elif mode == "Coordonnées Expéditeur":
                max_height = min(0.25 * h, bottom_limit - top_limit)
                if max_height <= 0:
                    pg["added"] = False
                    print("ko", bottom_limit, top_limit)
                    break
                pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                top = randint(top_limit, bottom_limit - img_h + 1)
                left = randint(0, int(0.5 * w - img_w) + 1)
            elif mode == "Coordonnées Destinataire":
                if h - max_bottom_crop - 10 > 0.2 * h and randint(0, 10) == 0:
                    pg_image = resize_max(pg_image, max_height=int(0.2 * h), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w - img_w)
                else:
                    max_height = min(0.25 * h, bottom_limit - top_limit)
                    if max_height <= 0:
                        pg["added"] = False
                        print("ko", bottom_limit, top_limit)
                        break
                    pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    if date_alone and w - date_right - img_w > 11:
                        top = randint(0, date_bottom - img_h + 1)
                        left = randint(max(int(0.5 * w), date_right + 10), w - img_w)
                    else:
                        top = randint(top_limit, bottom_limit - img_h + 1)
                        left = randint(int(0.5 * w), int(w - img_w) + 1)

            bottom = top + img_h
            right = left + img_w
            min_top_crop = min(top, min_top_crop)
            max_bottom_crop = max(bottom, max_bottom_crop)
            try:
                background[top:bottom, left:right, ...] = pg_image
            except:
                pg["added"] = False
                nb_lines = 0
            pg["coords"] = {
                "top": top,
                "bottom": bottom,
                "right": right,
                "left": left
            }
        sorted_pg = order_text_regions(paragraphs.values())
        for pg in sorted_pg:
            if "added" in pg.keys() and pg["added"]:
                pg_label = "\n".join(pg["lines"])
                mode = pg["type"]
                begin_token = SEM_START_TOKENS[mode]
                end_token = SEM_MATCHING_TOKENS[begin_token]
                page_labels["raw"].append(pg_label)
                page_labels["sem"].append(begin_token + pg_label + end_token)
        page_labels["raw"] = "\n".join(page_labels["raw"])
        page_labels["sem"] = "\n".join(page_labels["sem"])

        if crop:
            if min_top_crop > max_bottom_crop:
                print("KO - min > MAX")
            elif min_top_crop > h:
                print("KO - min > h")
            else:
                background = background[min_top_crop:max_bottom_crop]
        return background, page_labels["sem"] if self.add_sem_tokens else page_labels["raw"]

    def compute_sem_stat(self):
        stats = dict()
        for text_type in self.text_types:
            stats[text_type] = max(0.10, len(self.labels[text_type]) / self.num_samples)
        return stats
    
    def generate_paragraph(self, mode, mix=False, text_lines=None, num_all_lines=None):
        if text_lines is None:
            return self.get_paragraph_rimes(mode, mix, num_all_lines)
        # text from wikipedia
        num_text_lines = len(text_lines)
        if mode == "Corps de texte":
            min_ = min(num_text_lines, 5)
            max_ = min(num_text_lines+1, 30)
        else:
            min_ = min(num_text_lines, 2)
            max_ = min(num_text_lines+1, 8)
        num_lines = randint(min_, max_)
        label = "\n".join(text_lines[:num_lines])
        del text_lines[:num_lines]
        return {
            "type": mode,
            "label": label
        }

    def get_paragraph_rimes(self, mode="Corps de texte", mix=False, num_all_lines=1):
        while True:
            sample_idx = randint(0, len(self.labels[mode]))
            pg = self.labels[mode][sample_idx]
            if mode == "Corps de texte" and mix:
                return self.get_mix_paragraph_rimes(mode, max(1, num_all_lines-12))
            else:
                return {
                    "type": mode,
                    "label": pg
                }
    

    def get_mix_paragraph_rimes(self, mode="Corps de texte", num_lines=10):
        res = list()
        while len(res) != num_lines:
            sample_idx = randint(0, len(self.labels[mode]))
            pg = self.labels[mode][sample_idx]
            lines = pg.split("\n")
            res.append(lines[randint(0, len(lines))])
        return {
            "label": "\n".join(res),
            "type": mode,
        }

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

        padded_images = [pad_image_width_random(img, max_width, padding_value=DEFAULT_SYN_CONFIG["background_color_default"][0], max_pad_left_ratio=max_pad_left_ratio) for img in images]
        if self.overlap:
            return concatenate_lines_overlap(padded_images, padding_value=DEFAULT_SYN_CONFIG["background_color_default"][0])
        return np.concatenate(padded_images, axis=0)
