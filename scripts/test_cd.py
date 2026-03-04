"""
scripts/test_cd.py
===================
Evaluate a trained change-detection model on the test split.

Usage
-----
.. code-block:: bash

    python scripts/test_cd.py -c configs/levir.yaml -p test -gpu 0

Requires:

* ``path.resume_state`` — path to the pre-trained diffusion model directory
  (saved with :meth:`~src.models.DiffusionFeatureExtractor.save_pretrained`).
* ``path_cd.resume_state`` — path prefix for the CD model weights
  (e.g. ``experiments/.../checkpoint/best_cd_model``).
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_cd_dataloader, create_cd_dataset
from src.models import ChangeDetectionModel, DiffusionFeatureExtractor
from src.utils.logger import dict_to_nonedict, dict2str, parse, setup_logger
from src.utils.metrics import save_img, tensor2img
from src.utils.wandb_logger import WandbLogger


def _extract_features(extractor, data, opt, device):
    img_A = data["A"].to(device)
    img_B = data["B"].to(device)
    feats_A, feats_B = [], []
    for t in opt["model_cd"]["t"]:
        feats_A.append(extractor.extract_features(img_A, t))
        feats_B.append(extractor.extract_features(img_B, t))
    return feats_A, feats_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Test change-detection model.")
    parser.add_argument("-c", "--config", type=str, default="configs/levir.yaml")
    parser.add_argument("-p", "--phase", type=str, choices=["test"], default="test")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    args = parser.parse_args()

    opt = dict_to_nonedict(parse(args))

    setup_logger(None, opt["path"]["log"], "test", level=logging.INFO, screen=True)
    logger = logging.getLogger("base")
    logger_test = logging.getLogger("test")
    logger.info(dict2str(opt))

    wandb_logger = None
    if opt["enable_wandb"]:
        wandb_logger = WandbLogger(opt)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_set = create_cd_dataset(opt["datasets"]["test"], "test")
    test_loader = create_cd_dataloader(test_set, opt["datasets"]["test"], "test")
    opt["len_train_dataloader"] = 0
    opt["len_val_dataloader"] = 0
    logger.info("Test dataset ready (%d samples).", len(test_set))

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")

    resume_state = opt["path"].get("resume_state")
    if resume_state:
        extractor = DiffusionFeatureExtractor.from_pretrained(resume_state)
    else:
        logger.warning("No diffusion checkpoint specified; using random weights.")
        extractor = DiffusionFeatureExtractor.from_config(
            unet_config=opt["model"]["unet"],
            scheduler_config=opt["model"]["scheduler"],
        )
    extractor = extractor.to(device)
    extractor.unet.eval()
    for p in extractor.unet.parameters():
        p.requires_grad_(False)

    cd_model = ChangeDetectionModel(opt)
    cd_model.clear_cache()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    test_result_path = os.path.join(opt["path"]["results"], "test")
    os.makedirs(test_result_path, exist_ok=True)

    for step, test_data in enumerate(test_loader):
        feats_A, feats_B = _extract_features(extractor, test_data, opt, device)
        cd_model.feed_data(feats_A, feats_B, test_data)
        cd_model.test()
        cd_model.collect_running_batch_states()

        logs = cd_model.get_current_log()
        logger_test.info(
            "[Test] iter:[%d/%d] mF1:%.5f", step, len(test_loader), logs["running_acc"]
        )

        visuals = cd_model.get_current_visuals()
        pred_cm = visuals["pred_cm"].float() * 2.0 - 1.0
        gt_cm = visuals["gt_cm"].float() * 2.0 - 1.0

        img_A = tensor2img(test_data["A"], out_type=np.uint8, min_max=(-1, 1))
        img_B = tensor2img(test_data["B"], out_type=np.uint8, min_max=(-1, 1))
        pred_img = tensor2img(
            pred_cm.unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1)
        )
        gt_img = tensor2img(
            gt_cm.unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1)
        )

        save_img(img_A, os.path.join(test_result_path, f"img_A_{step}.png"))
        save_img(img_B, os.path.join(test_result_path, f"img_B_{step}.png"))
        save_img(pred_img, os.path.join(test_result_path, f"img_pred_cm_{step}.png"))
        save_img(gt_img, os.path.join(test_result_path, f"img_gt_cm_{step}.png"))

    cd_model.collect_epoch_states()
    logs = cd_model.get_current_log()
    msg = f"[Test summary] mF1={logs['epoch_acc']:.5f}\n"
    for k, v in logs.items():
        msg += f"  {k}: {v:.4e}\n"
    logger_test.info(msg)

    if wandb_logger:
        wandb_logger.log_metrics({
            "test/mF1": logs["epoch_acc"],
            "test/mIoU": logs.get("miou", 0),
            "test/OA": logs.get("acc", 0),
        })

    logger.info("Testing complete.")


if __name__ == "__main__":
    main()
