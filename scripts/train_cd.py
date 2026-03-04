"""
scripts/train_cd.py
====================
Fine-tune the change-detection head on top of a frozen pre-trained DDPM.

Usage
-----
.. code-block:: bash

    python scripts/train_cd.py -c configs/levir.yaml -p train -gpu 0

The script:

1. Loads a frozen :class:`~src.models.DiffusionFeatureExtractor` from the
   checkpoint given in ``path.resume_state``.
2. Creates a :class:`~src.models.ChangeDetectionModel`.
3. For each epoch extracts diffusion features at the timesteps in
   ``model_cd.t`` and trains the CD head.
4. Saves the best model (by mean F1 on the validation split).
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_cd_dataloader, create_cd_dataset
from src.models import ChangeDetectionModel, DiffusionFeatureExtractor
from src.utils.logger import dict_to_nonedict, dict2str, parse, setup_logger
from src.utils.metrics import save_img, tensor2img
from src.utils.wandb_logger import WandbLogger


def _extract_features(extractor, data, opt, device):
    """Helper: extract diffusion features for images A and B."""
    img_A = data["A"].to(device)
    img_B = data["B"].to(device)
    feats_A, feats_B = [], []
    for t in opt["model_cd"]["t"]:
        fa = extractor.extract_features(img_A, t)
        fb = extractor.extract_features(img_B, t)
        feats_A.append(fa)
        feats_B.append(fb)
    return feats_A, feats_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CD head on frozen DDPM features.")
    parser.add_argument("-c", "--config", type=str, default="configs/levir.yaml")
    parser.add_argument("-p", "--phase", type=str, choices=["train"], default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_eval", action="store_true")
    args = parser.parse_args()

    opt = dict_to_nonedict(parse(args))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
    setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])

    wandb_logger = None
    if opt["enable_wandb"]:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric("epoch")
        wandb.define_metric("training/train_step")
        wandb.define_metric("training/*", step_metric="train_step")
        wandb.define_metric("validation/val_step")
        wandb.define_metric("validation/*", step_metric="val_step")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_set = create_cd_dataset(opt["datasets"]["train"], "train")
    train_loader = create_cd_dataloader(train_set, opt["datasets"]["train"], "train")
    opt["len_train_dataloader"] = len(train_loader)

    val_set = create_cd_dataset(opt["datasets"]["val"], "val")
    val_loader = create_cd_dataloader(val_set, opt["datasets"]["val"], "val")
    opt["len_val_dataloader"] = len(val_loader)
    logger.info("Datasets ready.")

    # ------------------------------------------------------------------
    # Diffusion feature extractor (frozen)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")

    resume_state = opt["path"].get("resume_state")
    if resume_state:
        logger.info("Loading pre-trained diffusion model from %s", resume_state)
        extractor = DiffusionFeatureExtractor.from_pretrained(resume_state)
    else:
        logger.info("No pre-trained diffusion model specified; using random weights.")
        extractor = DiffusionFeatureExtractor.from_config(
            unet_config=opt["model"]["unet"],
            scheduler_config=opt["model"]["scheduler"],
        )

    extractor = extractor.to(device)
    extractor.unet.eval()
    for p in extractor.unet.parameters():
        p.requires_grad_(False)
    logger.info("Diffusion feature extractor ready (frozen).")

    # ------------------------------------------------------------------
    # Change-detection model
    # ------------------------------------------------------------------
    cd_model = ChangeDetectionModel(opt)
    logger.info("Change-detection model ready.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epoch = opt["train"]["n_epoch"]
    best_mF1 = 0.0
    train_opt = opt["train"]

    for epoch in range(n_epoch):
        cd_model.clear_cache()
        train_result_path = os.path.join(opt["path"]["results"], "train", str(epoch))
        os.makedirs(train_result_path, exist_ok=True)

        # --- Training ---
        lr = cd_model.optimizer.param_groups[0]["lr"]
        logger.info("[Epoch %d/%d] lr=%.7f", epoch, n_epoch - 1, lr)

        for step, train_data in enumerate(train_loader):
            feats_A, feats_B = _extract_features(extractor, train_data, opt, device)
            cd_model.feed_data(feats_A, feats_B, train_data)
            cd_model.optimize_parameters()
            cd_model.collect_running_batch_states()

            if step % train_opt["train_print_freq"] == 0:
                logs = cd_model.get_current_log()
                logger.info(
                    "[Train] epoch:[%d/%d] iter:[%d/%d] loss:%.5f mF1:%.5f",
                    epoch, n_epoch - 1, step, len(train_loader),
                    logs["l_cd"], logs["running_acc"],
                )

                visuals = cd_model.get_current_visuals()
                pred_cm = visuals["pred_cm"].float() * 2.0 - 1.0
                gt_cm = visuals["gt_cm"].float() * 2.0 - 1.0
                grid = torch.cat(
                    (
                        train_data["A"],
                        train_data["B"],
                        pred_cm.unsqueeze(1).repeat(1, 3, 1, 1),
                        gt_cm.unsqueeze(1).repeat(1, 3, 1, 1),
                    ),
                    dim=0,
                )
                grid_img = tensor2img(grid)
                save_img(
                    grid_img,
                    os.path.join(
                        train_result_path,
                        f"img_A_B_pred_gt_e{epoch}_b{step}.png",
                    ),
                )

        cd_model.collect_epoch_states()
        logs = cd_model.get_current_log()
        msg = f"[Train summary] epoch:[{epoch}/{n_epoch-1}] mF1={logs['epoch_acc']:.5f}\n"
        for k, v in logs.items():
            msg += f"  {k}: {v:.4e}  "
            tb_logger.add_scalar(k, v, step)
        logger.info(msg)

        if wandb_logger:
            wandb_logger.log_metrics({
                "training/mF1": logs["epoch_acc"],
                "training/mIoU": logs.get("miou", 0),
                "training/OA": logs.get("acc", 0),
                "training/train_step": epoch,
            })

        cd_model.clear_cache()
        cd_model.update_lr()

        # --- Validation ---
        if epoch % train_opt["val_freq"] == 0:
            val_result_path = os.path.join(opt["path"]["results"], "val", str(epoch))
            os.makedirs(val_result_path, exist_ok=True)

            for step, val_data in enumerate(val_loader):
                feats_A, feats_B = _extract_features(extractor, val_data, opt, device)
                cd_model.feed_data(feats_A, feats_B, val_data)
                cd_model.test()
                cd_model.collect_running_batch_states()

                if step % train_opt["val_print_freq"] == 0:
                    logs = cd_model.get_current_log()
                    logger.info(
                        "[Val] epoch:[%d/%d] iter:[%d/%d] mF1:%.5f",
                        epoch, n_epoch - 1, step, len(val_loader), logs["running_acc"],
                    )

            cd_model.collect_epoch_states()
            logs = cd_model.get_current_log()
            logger.info(
                "[Val summary] epoch:[%d/%d] mF1=%.5f",
                epoch, n_epoch - 1, logs["epoch_acc"],
            )
            for k, v in logs.items():
                tb_logger.add_scalar(k, v, step)

            if wandb_logger:
                wandb_logger.log_metrics({
                    "validation/mF1": logs["epoch_acc"],
                    "validation/mIoU": logs.get("miou", 0),
                    "validation/OA": logs.get("acc", 0),
                    "validation/val_step": epoch,
                })

            is_best = logs["epoch_acc"] > best_mF1
            if is_best:
                best_mF1 = logs["epoch_acc"]
                logger.info("[Val] New best model (mF1=%.5f).  Saving.", best_mF1)
            cd_model.save_checkpoint(epoch, is_best=is_best)
            cd_model.clear_cache()

        if wandb_logger:
            wandb_logger.log_metrics({"epoch": epoch})

    logger.info("Training complete.  Best validation mF1: %.5f", best_mF1)


if __name__ == "__main__":
    main()
