"""
scripts/train_diffusion.py
===========================
Pre-train an unconditional DDPM on remote-sensing images using the
HuggingFace ``diffusers`` library.

Usage
-----
.. code-block:: bash

    python scripts/train_diffusion.py -c configs/ddpm_train.yaml \\
        -p train -gpu 0

The script trains a :class:`diffusers.UNet2DModel` with
:class:`diffusers.DDPMScheduler` and periodically saves checkpoints using
:meth:`~src.models.DiffusionFeatureExtractor.save_pretrained`.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloader, create_image_dataset
from src.models import DiffusionFeatureExtractor
from src.utils.logger import dict_to_nonedict, dict2str, parse, setup_logger
from src.utils.metrics import save_img, tensor2img
from src.utils.wandb_logger import WandbLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDPM on remote-sensing images.")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/ddpm_train.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "-p", "--phase", type=str, choices=["train"], default="train",
        help="Training phase (only 'train' is supported; use DDPMPipeline for inference).",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    args = parser.parse_args()

    opt = dict_to_nonedict(parse(args))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
    logger = logging.getLogger("base")
    logger.info(dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])

    wandb_logger = None
    if opt["enable_wandb"]:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="train_step")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_set = create_image_dataset(opt["datasets"]["train"], "train")
    train_loader = create_dataloader(train_set, opt["datasets"]["train"], "train")
    logger.info("Dataset ready (%d samples).", len(train_set))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")

    extractor = DiffusionFeatureExtractor.from_config(
        unet_config=opt["model"]["unet"],
        scheduler_config=opt["model"]["scheduler"],
    ).to(device)

    # Optionally resume from a checkpoint
    if opt["path"].get("resume_state"):
        ckpt = opt["path"]["resume_state"]
        logger.info("Resuming from %s", ckpt)
        unet_path = os.path.join(ckpt, "unet")
        if os.path.isdir(unet_path):
            from diffusers import UNet2DModel
            extractor.unet = UNet2DModel.from_pretrained(unet_path).to(device)

    train_opt = opt["train"]
    opt_type = train_opt["optimizer"]["type"]
    lr = train_opt["optimizer"]["lr"]
    params = list(extractor.unet.parameters())

    if opt_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        raise NotImplementedError(f"Optimizer '{opt_type}' not implemented.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    current_step = 0
    n_iter = train_opt["n_iter"]
    save_freq = int(train_opt["save_checkpoint_freq"])
    val_freq = int(train_opt["val_freq"])
    print_freq = int(train_opt["print_freq"])
    ckpt_dir = opt["path"]["checkpoint"]

    extractor.unet.train()
    logger.info("Starting DDPM pre-training for %d iterations.", n_iter)

    epoch = 0
    while current_step < n_iter:
        epoch += 1
        for train_data in train_loader:
            current_step += 1
            if current_step > n_iter:
                break

            clean_images = train_data["img"].to(device)
            B = clean_images.shape[0]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                extractor.scheduler.config.num_train_timesteps,
                (B,),
                device=device,
                dtype=torch.long,
            )
            noisy_images = extractor.scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = extractor(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_step % print_freq == 0:
                msg = f"<epoch:{epoch:3d}, iter:{current_step:8,d}> loss: {loss.item():.4e}"
                logger.info(msg)
                tb_logger.add_scalar("train/loss", loss.item(), current_step)
                if wandb_logger:
                    wandb_logger.log_metrics(
                        {"train/loss": loss.item(), "train/train_step": current_step}
                    )

            if current_step % save_freq == 0:
                save_path = os.path.join(ckpt_dir, f"I{current_step}_E{epoch}")
                os.makedirs(save_path, exist_ok=True)
                extractor.save_pretrained(save_path)
                logger.info("Checkpoint saved → %s", save_path)

    # Save final model
    final_path = os.path.join(ckpt_dir, "best_model")
    os.makedirs(final_path, exist_ok=True)
    extractor.save_pretrained(final_path)
    logger.info("Training complete.  Final model saved → %s", final_path)


if __name__ == "__main__":
    main()
