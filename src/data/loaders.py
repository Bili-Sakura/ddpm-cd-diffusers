"""
DataLoader factory functions
============================
"""

from __future__ import annotations

import logging

import torch.utils.data

from .cd_dataset import CDDataset
from .image_dataset import ImageDataset

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    dataset_opt: dict,
    phase: str,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the DDPM pre-training dataset."""
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt.get("use_shuffle", True),
            num_workers=dataset_opt.get("num_workers", 4),
            pin_memory=True,
        )
    raise NotImplementedError(f"Phase '{phase}' not supported for image dataloader.")


def create_cd_dataloader(
    dataset: torch.utils.data.Dataset,
    dataset_opt: dict,
    phase: str,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the change-detection dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_opt["batch_size"],
        shuffle=dataset_opt.get("use_shuffle", False),
        num_workers=dataset_opt.get("num_workers", 4),
        pin_memory=True,
    )


def create_image_dataset(dataset_opt: dict, phase: str) -> ImageDataset:
    """Instantiate an :class:`~src.data.image_dataset.ImageDataset`."""
    ds = ImageDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt.get("data_len", -1),
    )
    logger.info("Dataset [%s - %s] created.", ds.__class__.__name__, dataset_opt["name"])
    return ds


def create_cd_dataset(dataset_opt: dict, phase: str) -> CDDataset:
    """Instantiate a :class:`~src.data.cd_dataset.CDDataset`."""
    ds = CDDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt.get("data_len", -1),
    )
    logger.info(
        "Dataset [%s - %s - %s] created.",
        ds.__class__.__name__,
        dataset_opt["name"],
        phase,
    )
    return ds
