"""
Data utilities
==============
Image I/O helpers, augmentation transforms and path utilities.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"}

# Reusable transforms
_totensor = torchvision.transforms.ToTensor()
_hflip = torchvision.transforms.RandomHorizontalFlip()


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1] in IMG_EXTENSIONS


def get_paths_from_images(path: str) -> List[str]:
    """Return sorted list of image file paths under *path* (recursive)."""
    assert os.path.isdir(path), f"{path!r} is not a valid directory"
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(dirpath, fname))
    assert images, f"{path!r} contains no valid image files"
    return images


def transform_augment(
    img: Image.Image,
    split: str = "val",
    min_max: tuple = (0, 1),
    res: int = 256,
) -> torch.Tensor:
    """Convert a PIL image to a normalised tensor with optional augmentation.

    Parameters
    ----------
    img:
        Input PIL image.
    split:
        ``"train"`` enables random horizontal flip and crop/resize.
    min_max:
        Output value range ``(lo, hi)``.
    res:
        Target resolution.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(C, res, res)`` in ``[min_max[0], min_max[1]]``.
    """
    tensor = _totensor(img)
    if split == "train":
        h = tensor.shape[1]
        if h < res:
            tensor = T.Resize(res)(tensor)
        elif h > res:
            tensor = T.RandomCrop(res)(tensor)
        tensor = _hflip(tensor)
    return tensor * (min_max[1] - min_max[0]) + min_max[0]


def transform_augment_cd(
    img: Image.Image,
    split: str = "val",
    min_max: tuple = (0, 1),
) -> torch.Tensor:
    """Minimal transform for change-detection datasets (no spatial aug)."""
    tensor = _totensor(img)
    return tensor * (min_max[1] - min_max[0]) + min_max[0]
