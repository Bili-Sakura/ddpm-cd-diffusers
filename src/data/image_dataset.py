"""
ImageDataset
============
PyTorch dataset for single (unpaired) remote-sensing images used during
DDPM pre-training (unconditional generation).
"""

from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

from .utils import get_paths_from_images, transform_augment


class ImageDataset(Dataset):
    """Loads single images for unconditional diffusion model pre-training.

    Parameters
    ----------
    dataroot:
        Directory containing image files (searched recursively).
    resolution:
        Target spatial resolution.
    split:
        ``"train"`` enables augmentation; anything else disables it.
    data_len:
        Maximum number of samples.  ``-1`` uses all found images.
    """

    def __init__(
        self,
        dataroot: str,
        resolution: int = 256,
        split: str = "train",
        data_len: int = -1,
    ) -> None:
        self.res = resolution
        self.split = split

        paths = get_paths_from_images(dataroot)
        dataset_len = len(paths)
        self.data_len = dataset_len if data_len <= 0 else min(data_len, dataset_len)
        self.paths = paths[: self.data_len]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> dict:
        img = Image.open(self.paths[index]).convert("RGB")
        img = transform_augment(img, split=self.split, min_max=(-1, 1), res=self.res)
        return {"img": img, "Index": index}
