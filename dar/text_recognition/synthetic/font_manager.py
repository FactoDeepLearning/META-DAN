import os
import numpy as np
import logging
from tqdm import tqdm
from fontTools import ttLib
from random import shuffle


class FontManager:

    # list of fonts with issues
    forbidden_fonts_keywords = [
        "Sumana",  # too much padding
    ]

    def __init__(self, params):
        self.valid_extensions = ["ttf", "TTF", "otf", "woff"]
        self.cache_folder = os.path.join(".cache", "fonts")
        self.output_folder = "Fonts"
        self.font_list_path = os.path.join(self.output_folder, "font_list.txt")

        # Do not print warnings of fonttools
        logger = logging.getLogger('fontTools')
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True

        assert len(list(os.listdir(self.output_folder))) > 0, "you must add at least one font to generate synthetic documents"

        self.max_num_fonts = params["syn_num_fonts"]
        self.all_fonts = self.get_font_paths()
        print(f"Number of loaded fonts: {len(self.all_fonts)}")

        print("Processing fonts...")
        self.charset_per_font = self.get_charset_per_font()
        print("Done!")
        self.valid_fonts = dict()


    @staticmethod
    def get_font_characters(font_path: str) -> list:
        """
        Get the list of available characters for given font path.
        """
        font = ttLib.TTFont(font_path)
        cmap = font['cmap'].getBestCmap()
        if cmap is None:
            return []
        return [chr(key) for key in cmap]

    def filter_fonts_with_tesseract(self):
        """
        Only preserve fonts that are rather recognizable by tesseract
        """
        for filename in tqdm(os.listdir(self.output_folder), desc="Filtering fonts"):
            font_path = os.path.join(self.output_folder, filename)
            if len(FontManager.get_font_characters(font_path)) < 10 or not self.is_recognizable_with_tesseract(font_path):
                os.remove(font_path)

    def filter_fonts_with_file(self) -> None:
        """
        Only preserve fonts declared in Fonts/font_list.txt
        """
        with open(self.font_list_path, "r") as f:
            valid_font_paths = f.read().split("\n")
        for filename in os.listdir(self.output_folder):
            font_path = os.path.join(self.output_folder, filename)
            if font_path not in valid_font_paths and self.is_font_path(font_path):
                os.remove(font_path)

    def is_font_path(self, path: str) -> bool:
        """
        Return True if given path corresponds to a font, by checking the extension.
        """
        return path.split(".")[-1] in self.valid_extensions
    
    def is_forbidden_font(self, font_path: str) -> bool:
        """
        Check if given font is known to be unusable or not
        """
        for keyword in self.forbidden_fonts_keywords:
            if keyword in font_path:
                return True
        return False

    def get_font_paths(self) -> list[str]:
        """
        Get all font paths by listing the Fonts/ folder
        """
        all_files = [os.path.join(self.output_folder, fn) for fn in os.listdir(self.output_folder)]
        all_files = [f for f in all_files if f.split(".")[-1] in self.valid_extensions]
        all_files = [f for f in all_files if not self.is_forbidden_font(f)]  # remove fonts for which we know they are problematic
        return all_files

    def export_font_list(self) -> None:
        """
        Export the list of considered valid fonts in Fonts/font_list.txt
        """
        font_paths = self.get_font_paths()
        np.random.shuffle(font_paths)
        with open(self.font_list_path, "w") as f:
            f.write("\n".join(font_paths))

    def get_charset_per_font(self) -> dict[str, list[str]]:
        """
        Compute a dictonary associating each font path to its available charset
        """
        charset_per_font = {}
        for font_path in self.all_fonts:
            charset_per_font[font_path] = FontManager.get_font_characters(font_path)
        return charset_per_font

    def get_valid_fonts(self, charset: list) -> list[str]:
        """
        Given a charset, returns the list of compatible font paths
        """
        return [font_path for font_path in self.all_fonts if self.is_valid_font(font_path, charset)]

    def is_valid_font(self, font_path: str, charset: list[str]) -> bool:
        """
        Return True if all character of the given charset is available in the font
        """
        charset = set(charset)
        if 'ﭐ' in charset:
            charset.remove('ﭐ')
            charset.add("ا")
        return charset.issubset(set(self.charset_per_font[font_path]))

    def get_random_font(self, text: str) -> str:
        """
        Randomly get a valid font for a given text
        """
        font = None
        all_fonts = self.all_fonts.copy()
        shuffle(all_fonts)
        for f in all_fonts:
            if self.is_valid_font(f, text):
                font = f
                break
        assert font is not None, f"No valid font for text: {text}"
        return font
