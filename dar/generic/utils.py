"""
Module defining generic utils functions (mainly random number generation and padding)
"""

import numpy as np
import torch
from torch.distributions.uniform import Uniform
import re
import cv2
import random
import os


def natural_sort(l: list[str]) -> list[str]:
    """
    Sort a list of string considering number natural sorting ("1", "2", "10") instead of ("1", "10", "2")
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def randint(low: int, high:int) -> int:
    """
    call torch.randint to preserve random among dataloader workers
    """
    return int(torch.randint(low, high, (1, )))


def rand() -> float:
    """
    call torch.rand to preserve random among dataloader workers
    """
    return float(torch.rand((1, )))


def rand_uniform(low: float, high: float) -> float:
    """
    call torch uniform to preserve random among dataloader workers
    """
    if low == high:
        return low
    return float(Uniform(low, high).sample())

def randint_normal(mean: float, std: float, min: float, max:float) -> float:
    """
    Generate random value following normal distribution with give mean and std, clamped between given min and max
    """
    return int(torch.clamp(torch.normal(mean, std, (1, )), min=min, max=max))


def pad_sequences_1D(data: list[np.ndarray], padding_value: int) -> np.ndarray:
    """
    Gather a list of 1D numpy array as a mini-batch representation by padding images of lower length.
    """
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)).astype(np.int32) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data


def resize_max(img: np.ndarray, max_width=None, max_height=None) -> np.ndarray:
    """
    Resize preserving ratio if width and/or height is bigger than given values.
    """
    if max_width is not None and img.shape[1] > max_width:
        ratio = max_width / img.shape[1]
        new_h = int(np.floor(ratio * img.shape[0]))
        new_w = int(np.floor(ratio * img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if max_height is not None and img.shape[0] > max_height:
        ratio = max_height / img.shape[0]
        new_h = int(np.floor(ratio * img.shape[0]))
        new_w = int(np.floor(ratio * img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def pad_images(data: list[np.ndarray], padding_value: int, padding_mode: str="br") -> np.ndarray:
    """
    Gather a list of images (as B numpy arrays of shape (Hi, Wi, C)) to a mini-batch representation of shape (B, Hmax, Wmax, C) using padding.
    mode: "br"/"tl"/"random" (where to put some padding: bottom-right, top-left, random)
    """
    x_lengths = [x.shape[0] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)
    if padding_value == "mean":
        padded_data = np.concatenate([generate_image_full_padding((1, longest_x, longest_y, data[0].shape[2]), padding_value, d) for d in data], axis=0)
    else:
        padded_data = np.ones((len(data), longest_x, longest_y, data[0].shape[2]), dtype=data[0].dtype) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        if padding_mode == "br":
            padded_data[i, :x_len, :y_len, ...] = data[i]
        elif padding_mode == "tl":
            padded_data[i, -x_len:, -y_len:, ...] = data[i]
        elif padding_mode == "random":
            xmax = longest_x - x_len
            ymax = longest_y - y_len
            xi = randint(0, xmax) if xmax >= 1 else 0
            yi = randint(0, ymax) if ymax >= 1 else 0
            padded_data[i, xi:xi+x_len, yi:yi+y_len, ...] = data[i]
        else:
            raise NotImplementedError("Undefined padding mode: {}".format(padding_mode))
    return padded_data


def pad_image(image: np.ndarray, padding_value: int, new_height: int=None, new_width: int=None, pad_width:int =None, pad_height:int =None, padding_mode: str="br", return_position: bool=False) -> np.ndarray:
    """
    Pad the given input image (a numpy array of shape (H, W, C))
    mode: "br"/"tl"/"random" (where to put some padding: bottom-right, top-left, random)
    """
    if pad_width is not None and new_width is not None:
        raise NotImplementedError("pad_with and new_width are not compatible")
    if pad_height is not None and new_height is not None:
        raise NotImplementedError("pad_height and new_height are not compatible")

    h, w, c = image.shape
    pad_width = pad_width if pad_width is not None else max(0, new_width - w) if new_width is not None else 0
    pad_height = pad_height if pad_height is not None else max(0, new_height - h) if new_height is not None else 0

    if not (pad_width == 0 and pad_height == 0):
        padded_image = generate_image_full_padding((h+pad_height, w+pad_width, c), padding_value, image, dtype=image[0].dtype)
        if padding_mode == "br":
            hi, wi = 0, 0
        elif padding_mode == "tl":
            hi, wi = pad_height, pad_width
        elif padding_mode == "random":
            hi = randint(0, pad_height) if pad_height >= 1 else 0
            wi = randint(0, pad_width) if pad_width >= 1 else 0
        else:
            raise NotImplementedError("Undefined padding mode: {}".format(padding_mode))
        padded_image[hi:hi + h, wi:wi + w, ...] = image
        output = padded_image
    else:
        hi, wi = 0, 0
        output = image

    if return_position:
        return output, [[hi, hi+h], [wi, wi+w]]
    return output


def pad_image_width_right(img: np.ndarray, new_width: int, padding_value: int) -> np.ndarray:
    """
    Pad img to right side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_right = generate_image_full_padding((h, pad_width, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([img, pad_right], axis=1)
    return img


def pad_image_width_left(img: np.ndarray, new_width: int, padding_value: int) -> np.ndarray:
    """
    Pad img to left side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_left = generate_image_full_padding((h, pad_width, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([pad_left, img], axis=1)
    return img


def pad_image_width_center(img: np.ndarray, new_width: int, padding_value: int) -> np.ndarray:
    """
    Pad img to left and right side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_left = generate_image_full_padding((h, pad_left, c), padding_value, img, dtype=img.dtype)
    pad_right = generate_image_full_padding((h, pad_right, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([pad_left, img, pad_right], axis=1)
    return img

def pad_image_width_random(img: np.ndarray, new_width: int, padding_value: int, max_pad_left_ratio: float=1.) -> np.ndarray:
    """
    Randomly pad img to left and right sides with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    max_pad_left = int(max_pad_left_ratio*pad_width)
    pad_left = randint(0, min(pad_width, max_pad_left)) if pad_width != 0 and max_pad_left > 0 else 0
    pad_right = pad_width - pad_left
    pad_left = generate_image_full_padding((h, pad_left, c), padding_value, img, dtype=img.dtype)
    pad_right = generate_image_full_padding((h, pad_right, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([pad_left, img, pad_right], axis=1)
    return img


def pad_image_height_random(img: np.ndarray, new_height: int, padding_value: int, max_pad_top_ratio: float=1.) -> np.ndarray:
    """
    Randomly pad img top and bottom sides with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_height = max((new_height - h), 0)
    max_pad_top = int(max_pad_top_ratio*pad_height)
    pad_top = randint(0, min(pad_height, max_pad_top)) if pad_height != 0 and max_pad_top > 0 else 0
    pad_bottom = pad_height - pad_top
    pad_top = generate_image_full_padding((pad_top, w, c), padding_value, img, dtype=img.dtype)
    pad_bottom = generate_image_full_padding((pad_bottom, w, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([pad_top, img, pad_bottom], axis=0)
    return img


def pad_image_height_bottom(img: np.ndarray, new_height: int, padding_value: int) -> np.ndarray:
    """
    Pad img to bottom side with padding value to reach new_height as height
    """
    h, w, c = img.shape
    pad_height = max((new_height - h), 0)
    pad_bottom = generate_image_full_padding((pad_height, w, c), padding_value, img, dtype=img.dtype)
    img = np.concatenate([img, pad_bottom], axis=0)
    return img


def generate_image_full_padding(shape: tuple[int, int, int], padding_value: int|str, image: np.ndarray=None, dtype=None) -> np.ndarray:
    """
    Generate a numpy array full of padding with given shape
    """
    fill_value = padding_value if padding_value != "mean" else np.mean(image)
    return np.full(shape, fill_value=fill_value, dtype=dtype)


def image_torch_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Revert ToTensor() operation
    """
    return (img*255).to(int).permute(1, 2, 0).numpy()


def list_split(list_: str, delimiter: str)-> list[str]:
    """
    Split a string into a a lsit of substring given some delimiter
    """
    res = list()
    current = list()
    for item in list_:
        if item == delimiter:
            res.append(current)
            current = list()
        else:
            current.append(item)
    if len(current) > 0:
        res.append(current)
    return res


def find_file_by_keyword(fold_path: str, keyword: str) -> str|None:
    """
    Given a fold path, find a filepath in that fold that contains the given keyword in its name.
    """
    for filename in os.listdir(fold_path):
        if keyword in filename:
            return os.path.join(fold_path, filename)
    return None


def set_deterministic(seed=0) -> None:
    """
    Make training deterministic
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True