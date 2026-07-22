"""
Module dedicated to generate synthetic documents following the double-column layout of READ 2016 dataset.
"""

from dar.generic.utils import randint, rand, rand_uniform
from dar.text_recognition.synthetic.synthetic_line import generate_typed_text_line_image_default_config
from dar.dataset_format.read2016 import SEM_MATCHING_TOKENS, SEM_START_TOKENS
import random
import cv2
import numpy as np
import json
import re


class READ2016Synthetic:

    text_types = ["annotation", "page-number", "body"]

    def __init__(self, font_manager, sem_label_path, double_page=False, add_sem_tokens=False):
        self.font_manager = font_manager
        self.double_page = double_page
        self.add_sem_tokens = add_sem_tokens
        self.layout_tokens = "".join(list(SEM_MATCHING_TOKENS.values()) + list(SEM_START_TOKENS.values()))
        self.labels = self.load_labels(sem_label_path)
        self.dataset_name = None

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
        for sample in labels["train"].values():
            text = "".join(sample["text"])
            for text_type in self.text_types:
                new_text_list = patterns_by_type[text_type].findall(text)
                new_text_list = [re.sub(f"[{self.layout_tokens}]", "", t) for t in new_text_list]
                labels_by_type[text_type].extend(new_text_list)
        return labels_by_type

    def get_random_line(self, mode):
        sample_index = randint(0, len(self.labels[mode]))
        sample = self.labels[mode][sample_index]
        lines = sample.split("\n")
        line_index = randint(0, len(lines))
        return lines[line_index]

    def get_printed_line_read_2016(self, mode):
        while True:
            line_label = self.get_random_line(mode)
            if (mode == "body" and len(line_label) > 5) or (mode == "annotation" and len(line_label) < 15 and not line_label.isdigit()):
                img = generate_typed_text_line_image_default_config(text=line_label, font_path=self.font_manager.get_random_font(line_label))
                return line_label, img

    def render(self, sample, num_lines_per_page=20, crop=False, text_lines=None):
        self.dataset_name = sample["dataset"]
        h, w, c = sample["img"].shape
        nb_pages = 2 if self.double_page else 1
        page_width = w // 2 if nb_pages == 2 else w
        pages = list()
        labels = list()
        for i in range(nb_pages):
            if nb_pages == 2:
                side = "left" if i == 0 else "right"
            else:
                side = sample["gt"]["side"]
            num_lines = num_lines_per_page[i]
            background = np.ones((h, page_width, c), dtype=sample["img"].dtype) * 255
            if i == 0 and nb_pages == 2:
                background[:, -2:, :] = 0
            coords = {
                "left": int(0.15 * page_width) if side == "left" else int(0.05 * page_width),
                "right": int(0.95 * page_width) if side == "left" else int(0.85 * page_width),
                "top": int(0.05 * h),
                "bottom": int(0.85 * h),
            }
            page, label, num_cols = self.render_page(background, side, coords, num_lines, crop)
            pages.append(page)
            labels.append(label["sem"] if self.add_sem_tokens else label["raw"])
        img = np.concatenate(pages, axis=1)
        label = "\n".join(labels)
        return img, label

    def render_page(self, background, side, coords, nb_lines, crop):
        two_column = False
        page_labels = {
            "raw": "",
            "begin": f"{SEM_START_TOKENS['page']}",
            "sem": f"{SEM_START_TOKENS['page']}",
        }
        area_top = 0 if crop else coords["top"]
        area_left = coords["left"]
        area_right = coords["right"]
        area_bottom = coords["bottom"]
        min_lines_per_body = 1
        max_lines_per_ann = 6

        num_page_text_label = str(randint(0, 1000))
        num_page_img = generate_typed_text_line_image_default_config(num_page_text_label,
                                                                     self.font_manager.get_random_font(num_page_text_label))

        if side == "left":
            background[area_top:area_top + num_page_img.shape[0],
            area_left:area_left + num_page_img.shape[1]] = num_page_img
        else:
            background[area_top:area_top + num_page_img.shape[0],
            area_right - num_page_img.shape[1]:area_right] = num_page_img
        for key in ["sem", "begin"]:
            page_labels[key] += f"{SEM_START_TOKENS['page-number']}"
        for key in page_labels.keys():
            page_labels[key] += num_page_text_label
        page_labels["sem"] += SEM_MATCHING_TOKENS[SEM_START_TOKENS['page-number']]
        area_top = area_top + num_page_img.shape[0]
        nb_lines -= 1
        if nb_lines > 0:
            area_top += randint(1, 25)
        ratio_ann = rand_uniform(0.6, 0.7)
        full_page = rand() > 0.85
        while nb_lines > 0:
            nb_body_lines = randint(min_lines_per_body, max(min_lines_per_body, nb_lines) + 1)
            if full_page:
                nb_lines = nb_body_lines
            max_ann_lines = max(0, min(nb_body_lines, nb_lines - nb_body_lines))
            body_labels = list()
            body_imgs = list()
            while nb_body_lines > 0:
                current_nb_lines = 1
                label, img = self.get_printed_line_read_2016("body")

                nb_body_lines -= current_nb_lines
                body_labels.append(label)
                body_imgs.append(img)
            nb_ann_lines = randint(0, min(max_lines_per_ann, max_ann_lines + 1))
            ann_labels = list()
            ann_imgs = list()
            while nb_ann_lines > 0:
                current_nb_lines = 1
                label, img = self.get_printed_line_read_2016("annotation")

                nb_ann_lines -= current_nb_lines
                ann_labels.append(label)
                ann_imgs.append(img)
            max_width_body = int(np.floor(ratio_ann * (area_right - area_left)))
            max_width_ann = area_right - area_left - max_width_body
            for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                for i in range(len(img_list)):
                    if img_list[i].shape[1] > max_width:
                        ratio = max_width / img_list[i].shape[1]
                        new_h = int(np.floor(ratio * img_list[i].shape[0]))
                        new_w = int(np.floor(ratio * img_list[i].shape[1]))
                        img_list[i] = cv2.resize(img_list[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            body_top = area_top
            body_height = 0
            i_body = 0
            for i, (label, img) in enumerate(zip(body_labels, body_imgs)):
                remaining_height = area_bottom - body_top
                if img.shape[0] > remaining_height:
                    nb_lines = 0
                    break
                background[body_top:body_top + img.shape[0],
                area_left + max_width_ann:area_left + max_width_ann + img.shape[1]] = img
                body_height += img.shape[0]
                body_top += img.shape[0]
                if full_page and i < len(body_labels) - 1:
                    interline_height = randint(int(0.9 * img.shape[0]), int(2 * img.shape[0]))
                    body_top += interline_height
                    body_height += interline_height
                nb_lines -= 1
                i_body += 1

            ann_height = int(np.sum([img.shape[0] for img in ann_imgs]))
            ann_top = area_top + randint(0, body_height - ann_height + 1) if ann_height < body_height else area_top
            largest_ann = max([a.shape[1] for a in ann_imgs]) if len(ann_imgs) > 0 else max_width_ann
            pad_ann = randint(0, max_width_ann - largest_ann + 1) if max_width_ann > largest_ann else 0

            ann_label_blocks = [list(), ]
            i_ann = 0
            ann_height = 0
            for (label, img) in zip(ann_labels, ann_imgs):
                remaining_height = body_top - ann_top
                if img.shape[0] > remaining_height:
                    break
                background[ann_top:ann_top + img.shape[0], area_left + pad_ann:area_left + pad_ann + img.shape[1]] = img
                ann_height += img.shape[0]
                ann_top += img.shape[0]
                nb_lines -= 1
                two_column = True
                ann_label_blocks[-1].append(ann_labels[i_ann])
                i_ann += 1
                if randint(0, 10) == 0:
                    ann_label_blocks.append(list())
                    ann_top += randint(0, max(15, body_top - ann_top - 20))

            area_top = area_top + max(ann_height, body_height)
            if nb_lines > 0:
                area_top += randint(25, 100)

            ann_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            for ann_label_block in ann_label_blocks:
                if len(ann_label_block) > 0:
                    for key in ["sem", "begin"]:
                        ann_full_labels[key] += SEM_START_TOKENS['annotation']
                    ann_full_labels["raw"] += "\n"
                    for key in ann_full_labels.keys():
                        ann_full_labels[key] += "\n".join(ann_label_block)
                    ann_full_labels["sem"] += SEM_MATCHING_TOKENS[SEM_START_TOKENS['annotation']]

            body_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            if i_body > 0:
                for key in ["sem", "begin"]:
                    body_full_labels[key] += SEM_START_TOKENS['body']
                body_full_labels["raw"] += "\n"
                for key in body_full_labels.keys():
                    body_full_labels[key] += "\n".join(body_labels[:i_body])
                body_full_labels["sem"] += SEM_MATCHING_TOKENS[SEM_START_TOKENS['body']]


            section_labels = dict()
            for key in ann_full_labels.keys():
                section_labels[key] = ann_full_labels[key] + body_full_labels[key]
            for key in section_labels.keys():
                if section_labels[key] != "":
                    if key in ["sem", "begin"]:
                        section_labels[key] = f"{SEM_START_TOKENS['section']}" + section_labels[key]
                    if key == "sem":
                        section_labels[key] = section_labels[key] + SEM_MATCHING_TOKENS[SEM_START_TOKENS['section']]
            for key in page_labels.keys():
                page_labels[key] += section_labels[key]

        if crop:
            background = background[:area_top]

        page_labels["sem"] += SEM_MATCHING_TOKENS[SEM_START_TOKENS['page']]

        for key in page_labels.keys():
            page_labels[key] = page_labels[key].strip()

        return [background, page_labels, 2 if two_column else 1]

    def get_n_consecutive_lines_read_2016(self, n=1, mode="body"):
        while True:
            sample = self.samples[randint(0, len(self))]
            paragraphs = list()
            for page in sample["pages_label"]:
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    if ((mode == "body" and pg["mode"] == "body") or
                        (mode == "ann" and pg["mode"] == "annotation")) and len(pg["lines"]) >= n:
                        line_idx = randint(0, len(pg["lines"]) - n + 1)
                        lines = pg["lines"][line_idx:line_idx + n]
                        label = "\n".join([l["text"] for l in lines])
                        top = min([l["top"] for l in lines])
                        bottom = max([l["bottom"] for l in lines])
                        left = min([l["left"] for l in lines])
                        right = max([l["right"] for l in lines])
                        img = sample["img"][top:bottom, left:right]
                        return label, img
