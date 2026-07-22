"""
Module defining useful functions for visualizations.
"""

import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import cv2


def add_custom_color_maps() -> None:
    """
    Add several matplotlib custom color maps with transparency.
    """
    color_names = ["red", "green", "blue", "yellow", "cyan", "pink"]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    for color, color_name in zip(colors, color_names):
        add_color_map_alpha(color, color_name)
        add_color_map_alpha_bar(color, color_name)


def add_color_map_alpha(color: list[int, int, int], name: str, num=256) -> None:
    """
    Add matplotlib custom color map with transparency for given original color.
    """
    colors = np.zeros((num, 4))
    colors[:, -1] = np.linspace(0.0, 0.75, num)
    for i in range(len(color)):
        colors[:, i] = color[i] / 255
    map_object = LinearSegmentedColormap.from_list(name='{}_alpha'.format(name), colors=colors)
    matplotlib.colormaps.register(cmap=map_object)


def add_color_map_alpha_bar(color: list[int, int, int], name: str) -> None:
    """
    Add matplotlib custom color map with transparency for given original color (for bar plots).
    """
    transparency_ticks = 50
    color = np.array(color) / 255
    colors = list()
    for a in range(transparency_ticks):
        temp = np.array([
            1 if color[0] == 1 else 1-a/transparency_ticks,
            1 if color[1] == 1 else 1-a/transparency_ticks,
            1 if color[2] == 1 else 1-a/transparency_ticks,
        ])
        colors.append(temp)
    cmap = ListedColormap(name=f"{name}_alpha_bar", colors=colors)
    matplotlib.colormaps.register(cmap=cmap)


def format_attention_weights(attn_weights: list[np.ndarray], img_size: list[int, int], mode=cv2.INTER_LINEAR) -> list[np.ndarray]:
    """
    Format the attention weights to fit the original image.
    Each attention weight maps must be of size (L, H, W) or (H, W)
    """
    new_attn_weights = []
    for i in range(len(attn_weights)):
        if len(attn_weights[0].shape) == 3:
            new_attn_weights.append(np.concatenate([np.expand_dims(cv2.resize(attn_weights[i][j], img_size[::-1], interpolation=mode), axis=0) for j in range(attn_weights[i].shape[0])], axis=0))
        else:
            new_attn_weights.append(cv2.resize(attn_weights[0], img_size[::-1], interpolation=mode))
    return new_attn_weights


def format_display_tokens(token: str) -> str:
    """
    If the given token is invisible, translate it to a markup, return the input otherwise.
    """
    token_map = {
        " ": "<space>",
        "\n": "<linebreak>",
    }
    return token_map[token] if token in token_map else token


def format_special_tokens(token: str) -> str:
    """
    If the given token is invisible, replace it by empty string in final prediction.
    """
    special_tokens_to_remove = ["<end>"]
    return "" if token in special_tokens_to_remove else token


def format_list_special_tokens(tokens: list[str]) -> list[str]:
    """
    If the given tokens are invisible, replace them by empty string in final prediction.
    """
    return [format_special_tokens(t) for t in tokens]