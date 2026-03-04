"""
Image metrics and visualisation helpers.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def tensor2img(
    tensor: torch.Tensor,
    out_type=np.uint8,
    min_max: tuple = (-1, 1),
) -> np.ndarray:
    """Convert a float tensor to a NumPy image array.

    Parameters
    ----------
    tensor:
        4-D ``(B, C, H, W)``, 3-D ``(C, H, W)`` or 2-D ``(H, W)`` tensor.
    out_type:
        Output dtype (``np.uint8`` by default).
    min_max:
        Input value range used for normalisation to ``[0, 1]``.

    Returns
    -------
    np.ndarray
        HWC uint8 image or HW array for single-channel inputs.
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = np.transpose(tensor.numpy(), (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f"Only 2-D, 3-D or 4-D tensors supported; got {n_dim}-D.")

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def save_img(img: np.ndarray, img_path: str, mode: str = "RGB") -> None:
    """Save an HWC RGB NumPy array to disk."""
    os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
