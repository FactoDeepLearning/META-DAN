from numpy.random import choice
import numpy as np

from dar.text_recognition.synthetic.synthetic_line import generate_typed_text_line_image_from_params
from dar.generic.utils import resize_max, rand, randint, pad_image_width_left, pad_image_width_right, pad_image_width_random, pad_image_width_center, rand_uniform

DEFAULT_SYN_CONFIG_BLOCK = {
            "background_color_default": (255, 255, 255),
            "text_color_default": (0, 0, 0),
            "font_size_min": 35,
            "font_size_max": 45,
            "color_mode": "RGB",
            "padding_left_ratio_min": 0.0,
            "padding_left_ratio_max": 0.0,
            "padding_right_ratio_min": 0.0,
            "padding_right_ratio_max": 0.0,
            "padding_top_ratio_min": 0.0,
            "padding_top_ratio_max": 0.0,
            "padding_bottom_ratio_min": 0.0,
            "padding_bottom_ratio_max": 0.0,
        }


class SyntheticComponent:
    pass


class SyntheticPage(SyntheticComponent):

    def __init__(self, height, width, font_manager, syn_config=None, ratio_num_cols=[0.8, 0.2]):
        self.width, self.height = width, height
        self.font_manager = font_manager
        self.syn_config = DEFAULT_SYN_CONFIG_BLOCK if syn_config is None else syn_config
        self.padding_value = self.syn_config["background_color_default"][0]
        self.page_config = {
            "min_section_height": 40,
            "min_section_padding": 25,
            "max_section_padding": 75,
            "max_width_padding": 100,
        }
        self.image = np.full((self.height, self.width, 3), dtype=np.uint8, fill_value=self.padding_value)
        self.text_lines = list()
        self.ratio_num_cols = ratio_num_cols

    def render(self, text, set_name, word_sep=" "):
        remaining_height = self.height
        if self.height > 200:
            remaining_height -= randint(0, 20)
        while len(text) > 0 and remaining_height > self.page_config["min_section_height"]:
            num_words = len(text.split(word_sep)) if word_sep else len(text)
            if len(text) <= 0 or num_words <= 0:
                continue
            padding_width_section = randint(1, self.page_config["max_width_padding"]+1)
            left = randint(0, padding_width_section)
            section_height = randint(self.page_config["min_section_height"], remaining_height)
            num_columns_section = np.random.choice(list(range(1, len(self.ratio_num_cols)+1)), p=self.ratio_num_cols)
            current_section = SyntheticSection(section_height, self.width-padding_width_section, self.font_manager, self.syn_config, num_columns_section)
            ratio_height = section_height / remaining_height
            max_words = max(int(min(ratio_height*num_words+0.1*num_words, num_words+1)), 2)
            min_words = min(max(1, int(ratio_height*num_words-0.1*num_words)), max_words-1)
            num_words_section = randint(min_words, max_words)
            section_words = text.split(word_sep) if word_sep else text
            section_words = section_words[:num_words_section]
            section_text = word_sep.join(section_words)
            current_section.render(section_text, set_name, word_sep=word_sep)
            self.text_lines += current_section.text_lines
            self.image[self.height-remaining_height:self.height-remaining_height+section_height, left:left+current_section.image.shape[1]] = current_section.image
            remaining_height -= section_height
            text = text[len(word_sep.join(current_section.text_lines)):].strip()

            if remaining_height > self.page_config["min_section_padding"]:
                padding = randint(self.page_config["min_section_padding"], min(remaining_height, self.page_config["max_section_padding"]))
                remaining_height -= padding


class SyntheticSection(SyntheticComponent):
    def __init__(self, height, width, font_manager, syn_config=None, num_columns=1):
        self.section_config = {
            "pad_min_columns": 10,
            "pad_max_columns": 50,
            "min_column_width_ratio": 0.2,
        }

        self.font_manager = font_manager
        self.syn_config = DEFAULT_SYN_CONFIG_BLOCK if syn_config is None else syn_config
        self.padding_value = self.syn_config["background_color_default"][0]
        self.width, self.height = width, height
        self.image = np.full((self.height, self.width, 3), dtype=np.uint8, fill_value=self.padding_value)
        self.num_columns = num_columns
        self.blocks = list()
        self.text_lines = list()

    def render(self, text, set_name, word_sep=" "):
        if self.num_columns == 1:
            self.blocks.append(SyntheticBlock(width=self.width,
                                              height=self.height,
                                              font_manager=self.font_manager,
                                              syn_config=self.syn_config))
            self.blocks[-1].render(text, set_name, word_sep=word_sep)
            self.image = self.blocks[-1].image
            self.text_lines = self.blocks[-1].text_lines
        else:
            last_width = 0
            padding_spaces = [randint(self.section_config["pad_min_columns"], self.section_config["pad_max_columns"]) for _ in range(self.num_columns-1)]
            remaining_space = self.width-np.sum(padding_spaces)
            min_width_per_column = int(np.floor(self.section_config["min_column_width_ratio"] * self.width))
            for i in range(self.num_columns):
                num_words = len(text.split(word_sep) if word_sep else text)
                if len(text) <= 0 or num_words <= 0:
                    continue
                min_space_for_remaining_columns = (self.num_columns-i) * min_width_per_column
                max_available_space = int(np.floor(remaining_space-min_space_for_remaining_columns))
                column_width = randint(min_width_per_column, min_width_per_column+max_available_space+1)
                current_block = SyntheticBlock(width=column_width,
                                                  height=self.height,
                                                  font_manager=self.font_manager,
                                                  syn_config=self.syn_config)
                mean_words_per_column = num_words/(self.num_columns-i)
                min_words = max(1, int(np.floor(mean_words_per_column-0.1*num_words)))
                max_words = max(min(int(mean_words_per_column+0.1*num_words), num_words+1), 2)
                column_num_words = randint(min_words, max_words)
                if word_sep:
                    column_text = word_sep.join(text.split(word_sep)[:column_num_words])
                else:
                    column_text = text[:column_num_words]

                current_block.render(column_text, set_name, force_top=i==0, word_sep=word_sep)
                text = text[len(word_sep.join(current_block.text_lines)):].strip()
                column_image = current_block.image
                self.image[:, last_width:last_width+column_image.shape[1]] = column_image
                self.text_lines += current_block.text_lines
                last_width = last_width+column_image.shape[1]
                remaining_space -= column_image.shape[1]
                if i < self.num_columns-1:
                    last_width += padding_spaces[i]
                self.blocks.append(current_block)


class SyntheticBlock(SyntheticComponent):
    def __init__(self, width, height, font_manager, block_config=None, syn_config=None):
        self.block_config = {
            "proba_text_alignment": {
                "left": 0.65,
                "center": 0.2,
                "right": 0.1,
                "random": 0.05
            },
            "proba_horizontal_alignment_in_bloc": {
                "left": 0.35,
                "center": 0.3,
                "right": 0.3,
                "random": 0.05
            },
            "proba_vertical_alignment_in_bloc": {
                "top": 0.35,
                "center": 0.3,
                "bottom": 0.3,
                "random": 0.05
            },
            "line_overlap": {
                "proba": 0.2,
                "max_height_ratio": 0.4,
                "min_height_ratio": 0.15
            },
            "space_between_lines": {
                "proba": 0.1,
                "max_space_ratio": 3,
            },
            "min_pixels_per_char": (40, 15)

        } if block_config is None else block_config

        self.syn_config = DEFAULT_SYN_CONFIG_BLOCK if syn_config is None else syn_config

        self.padding_value = self.syn_config["background_color_default"][0]
        self.width, self.height = width, height
        self.image = np.full((self.height, self.width, 3), dtype=np.uint8, fill_value=self.padding_value)
        self.max_line_width = self.width
        self.font_manager = font_manager
        self.text_lines = None

    def render(self, text_or_text_lines, set_name, force_top=False, word_sep=" "):
        if isinstance(text_or_text_lines, list):
            text_lines = text_or_text_lines
        else:
            text_lines = self.split_text_in_lines(text_or_text_lines, word_sep=word_sep)
        try:
            self.max_line_height = self.height//len(text_lines)
        except:
            print("ok")
        text_line_images = list()
        for text_line in text_lines:
            line_image = generate_typed_text_line_image_from_params(text_line, self.syn_config, self.font_manager.get_random_font(text_line), max_width=self.width)
            if line_image.shape[0] > self.max_line_height:
                line_image = resize_max(line_image, max_height=self.max_line_height)
            text_line_images.append(line_image)
        if len(text_lines) == 0:
            self.text_lines = [""]
            return

        max_width = np.max([im.shape[1] for im in text_line_images])
        text_alignment = choice(list(self.block_config["proba_text_alignment"].keys()), p=list(self.block_config["proba_text_alignment"].values()))
        stacked_lines = SyntheticBlock.stack_lines(line_images=text_line_images, max_height=self.height, overlap_config=self.block_config["line_overlap"], alignment=text_alignment, padding_value=self.padding_value, spacing_config=self.block_config["space_between_lines"])

        block_horizontal_alignment = choice(list(self.block_config["proba_horizontal_alignment_in_bloc"].keys()), p=list(self.block_config["proba_horizontal_alignment_in_bloc"].values()))
        init_pixel_width = SyntheticBlock.get_init_pixel_block_horizontal_alignment(self.width, max_width, block_horizontal_alignment)
        if force_top:
            init_pixel_height = 0
        else:
            block_vertical_alignment = choice(list(self.block_config["proba_vertical_alignment_in_bloc"].keys()), p=list(self.block_config["proba_vertical_alignment_in_bloc"].values()))
            init_pixel_height = SyntheticBlock.get_init_pixel_block_vertical_alignment(self.height, stacked_lines.shape[0], block_vertical_alignment)

        self.image[init_pixel_height:init_pixel_height+stacked_lines.shape[0], init_pixel_width:init_pixel_width+stacked_lines.shape[1]] = stacked_lines
        self.text_lines = text_lines

    def split_text_in_lines(self, text, word_sep=" "):
        max_lines = int(np.floor(self.height / self.block_config["min_pixels_per_char"][0]))
        max_char_per_line = int(np.floor(self.width / self.block_config["min_pixels_per_char"][1]))
        words = text.split(word_sep) if word_sep else list(text)
        random_newline_proba = 0.98
        lines = list()
        line = list()
        for word in words:
            if len(word) <= 0:
                continue
            if len(line) == 0:
                line.append(word)
                continue
            if len(word) + 1 + len(word_sep.join(line)) <= max_char_per_line and rand() < random_newline_proba:
                line.append(word)
                continue
            else:
                lines.append(word_sep.join(line))
                line = [word, ]
        if len(line) > 0:
            lines.append(word_sep.join(line))
        return lines[:max_lines]





    @staticmethod
    def get_init_pixel_block_horizontal_alignment(block_width, max_line_width, alignment):
        if alignment == "left":
            return 0

        if alignment == "right":
            return block_width-max_line_width

        if alignment == "center":
            return int(np.floor((block_width - max_line_width) / 2))

        if alignment == "random":
            max_shift = block_width-max_line_width
            return randint(0, max_shift+1)

    @staticmethod
    def get_init_pixel_block_vertical_alignment(block_height, height, alignment):
        if alignment == "top":
            return 0

        if alignment == "bottom":
            return block_height-height

        if alignment == "center":
            return int(np.floor((block_height - height) / 2))

        if alignment == "random":
            max_shift = block_height-height
            return randint(0, max_shift+1)

    @staticmethod
    def stack_lines(line_images, max_height, overlap_config=None, alignment="left", padding_value=0, spacing_config=None):
        max_width = np.max([im.shape[1] for im in line_images])
        raw_block_height = np.sum([im.shape[0] for im in line_images])
        remaining_space = max_height - raw_block_height
        line_images = [SyntheticBlock.pad_line_width_alignment(l, max_width, alignment, padding_value) for l in line_images]
        bloc = line_images[0]
        for line in line_images[1:]:
            overlap = 0
            space = 0 
            if rand() < spacing_config["proba"]:
                max_spacing_ratio = spacing_config["max_space_ratio"]
                max_space = min(max_spacing_ratio*line.shape[0], remaining_space)
                if max_space > 0:
                    space = randint(0, max_space+1)
                    bloc = np.concatenate([bloc, np.full((space, max_width, 3), fill_value=padding_value)])
                    remaining_space -= space
            elif rand() < overlap_config["proba"]:
                min_overlap = int(np.floor(overlap_config["min_height_ratio"] * line.shape[0]))
                max_overlap = int(np.floor(overlap_config["max_height_ratio"] * line.shape[0]))
                max_overlap = min(int(np.floor(overlap_config["max_height_ratio"]*bloc.shape[0])), max_overlap)
                overlap = randint(min_overlap, max_overlap) if max_overlap > min_overlap else 0
            new_bloc = np.full((bloc.shape[0]+line.shape[0]-overlap, max_width, 3), fill_value=padding_value)
            try:
                new_bloc[:bloc.shape[0]] = bloc
                line_mask = line!=padding_value
                new_bloc[-line.shape[0]:][line_mask] = line[line_mask]
            except:
                print(f"Error: {bloc.shape} {line.shape} overlap")
                import sys
                sys.exit(0)
            bloc = new_bloc
        return bloc

    @staticmethod
    def pad_line_width_alignment(line, width, alignment, padding_value=0):
        if line.shape[1] == width:
            return line
        if alignment == "left":
            return pad_image_width_right(img=line, new_width=width, padding_value=padding_value)
        if alignment == "right":
            return pad_image_width_left(img=line, new_width=width, padding_value=padding_value)
        if alignment == "center":
            return pad_image_width_center(img=line, new_width=width, padding_value=padding_value)
        if alignment == "random":
            return pad_image_width_random(img=line, new_width=width, padding_value=padding_value)
        

def concatenate_lines_overlap(line_images, padding_value, max_height_percentage_overlap=0.4):
    H, W, C = line_images[0].shape
    result = np.full((np.sum([img.shape[0] for img in line_images]), W, C), fill_value=padding_value, dtype=line_images[0].dtype)
    end_pos = 0
    last_height = 0
    shift_percent = rand_uniform(0, max_height_percentage_overlap)
    for line_image in line_images:
        shift = int(shift_percent * last_height)
        start_pos = end_pos - shift
        last_height = line_image.shape[0]
        end_pos = start_pos+last_height 
        mask = line_image != padding_value
        result[start_pos:end_pos][mask] = line_image[mask]
    return result
