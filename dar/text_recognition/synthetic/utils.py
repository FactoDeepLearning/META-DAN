from fontTools.ttLib import TTFont
import os
from dar.generic.utils import randint


def get_valid_fonts(alphabet=None):
    valid_fonts = list()
    for fold_detail in os.walk("./Fonts"):
        if fold_detail[2]:
            for font_name in fold_detail[2]:
                font_path = os.path.join(fold_detail[0], font_name)
                to_add = True
                if alphabet is not None:
                    for char in alphabet:
                        if not char_in_font(char, font_path):
                            to_add = False
                            break
                    if to_add:
                        valid_fonts.append(font_path)
                else:
                    valid_fonts.append(font_path)
    return valid_fonts


def char_in_font(unicode_char: str, font_path: str) -> bool:
    """
    Check if a given char is available in a given font
    """
    with TTFont(font_path) as font:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
    return False


def generate_colors_from_config(config: dict) -> tuple[tuple[int], tuple[int]]:
    background_color = config["background_color_default"]
    eps = randint(-config["background_color_eps"], config["background_color_eps"] + 1)
    background_color = tuple([min(255, max(0, c + eps)) for c in background_color])

    text_color = config["text_color_default"]
    eps = randint(-config["text_color_eps"], config["text_color_eps"] + 1)
    text_color = tuple([min(255, max(0, c + eps)) for c in text_color])
    return background_color, text_color