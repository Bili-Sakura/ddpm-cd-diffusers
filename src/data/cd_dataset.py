"""
CDDataset
=========
PyTorch dataset for bi-temporal change-detection benchmarks.

Expected directory layout::

    dataroot/
        A/          ← pre-change RGB images
        B/          ← post-change RGB images
        label/      ← binary change labels (single-channel PNG)
        list/
            train.txt
            val.txt
            test.txt
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import transform_augment_cd

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = "B"
LIST_FOLDER_NAME = "list"
ANNOT_FOLDER_NAME = "label"


def _load_img_name_list(dataset_path: str) -> np.ndarray:
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class CDDataset(Dataset):
    """Loads paired (pre, post) images and binary change labels.

    Parameters
    ----------
    dataroot:
        Root directory of the dataset (see module docstring for layout).
    resolution:
        Target spatial resolution (currently informational; resizing is done
        in the transform).
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    data_len:
        Maximum number of samples.  ``-1`` uses the full split.
    """

    def __init__(
        self,
        dataroot: str,
        resolution: int = 256,
        split: str = "train",
        data_len: int = -1,
    ) -> None:
        self.root_dir = dataroot
        self.split = split
        self.res = resolution

        list_path = os.path.join(dataroot, LIST_FOLDER_NAME, f"{split}.txt")
        self.img_names = _load_img_name_list(list_path)
        dataset_len = len(self.img_names)
        self.data_len = dataset_len if data_len <= 0 else min(data_len, dataset_len)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> dict:
        name = self.img_names[index % self.data_len]
        img_A = Image.open(os.path.join(self.root_dir, IMG_FOLDER_NAME, name)).convert("RGB")
        img_B = Image.open(os.path.join(self.root_dir, IMG_POST_FOLDER_NAME, name)).convert("RGB")
        lbl = Image.open(os.path.join(self.root_dir, ANNOT_FOLDER_NAME, name)).convert("RGB")

        img_A = transform_augment_cd(img_A, split=self.split, min_max=(-1, 1))
        img_B = transform_augment_cd(img_B, split=self.split, min_max=(-1, 1))
        lbl = transform_augment_cd(lbl, split=self.split, min_max=(0, 1))
        if lbl.dim() > 2:
            lbl = lbl[0]

        return {"A": img_A, "B": img_B, "L": lbl, "Index": index}
