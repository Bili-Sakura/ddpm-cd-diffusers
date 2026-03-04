"""
ChangeDetectionModel
====================
Wraps a :class:`~src.models.cd_head.ChangeDetectionHead` with training and
testing logic: loss, optimizer, LR scheduler, checkpointing and metrics.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cd_head import ChangeDetectionHead, get_in_channels
from ..utils.metric_tools import ConfuseMatrixMeter

logger = logging.getLogger(__name__)


def _get_lr_scheduler(optimizer: torch.optim.Optimizer, train_opt: dict):
    """Build an LR scheduler from the training config section."""
    from torch.optim import lr_scheduler

    policy = train_opt["scheduler"]["lr_policy"]
    n_epoch = train_opt["n_epoch"]

    if policy == "linear":
        lam = lambda epoch: 1.0 - epoch / float(n_epoch + 1)  # noqa: E731
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
    elif policy == "step":
        n_steps = train_opt["scheduler"].get("n_steps", 3)
        step_size = max(n_epoch // n_steps, 1)
        gamma = train_opt["scheduler"].get("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise NotImplementedError(f"LR policy '{policy}' is not implemented.")


class ChangeDetectionModel:
    """Manages the change-detection head: optimiser, loss, metrics and I/O.

    Parameters
    ----------
    opt:
        Full configuration dictionary (as produced by
        :func:`~src.utils.logger.parse`).
    """

    def __init__(self, opt: dict) -> None:
        self.opt = opt
        self.device = torch.device(
            "cuda" if opt.get("gpu_ids") is not None else "cpu"
        )
        self.phase = opt["phase"]

        cd_opt = opt["model_cd"]
        model_opt = opt["model"]

        # Build the change-detection head
        # The CD head uses features from the UNet's up_blocks[0:-1] (all but
        # the last no-upsample block).  The channel counts for those levels
        # equal block_out_channels[1:] — the first entry corresponds to the
        # skipped final up_block.  See DiffusionFeatureExtractor.block_out_channels.
        block_out_channels = tuple(model_opt["unet"]["block_out_channels"])[1:]
        self.net = ChangeDetectionHead(
            feat_scales=cd_opt["feat_scales"],
            block_out_channels=block_out_channels,
            out_channels=cd_opt["out_channels"],
            img_size=cd_opt["output_cm_size"],
            time_steps=cd_opt["t"],
        ).to(self.device)

        # Loss
        loss_type = cd_opt.get("loss_type", "ce")
        if loss_type == "ce":
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' not implemented.")

        self.log_dict: Dict = OrderedDict()
        self.running_metric = ConfuseMatrixMeter(n_class=cd_opt["out_channels"])

        if self.phase == "train":
            self.net.train()
            train_opt = opt["train"]
            opt_type = train_opt["optimizer"]["type"]
            lr = train_opt["optimizer"]["lr"]
            params = list(self.net.parameters())

            if opt_type == "adam":
                self.optimizer = torch.optim.Adam(params, lr=lr)
            elif opt_type == "adamw":
                self.optimizer = torch.optim.AdamW(params, lr=lr)
            else:
                raise NotImplementedError(f"Optimizer '{opt_type}' not implemented.")

            self.lr_scheduler = _get_lr_scheduler(self.optimizer, train_opt)
        else:
            self.net.eval()

        self._load_checkpoint()

        # Dataloader lengths (set externally after dataset creation)
        self.len_train_dataloader: int = opt.get("len_train_dataloader", 0)
        self.len_val_dataloader: int = opt.get("len_val_dataloader", 0)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _checkpoint_dir(self) -> str:
        return self.opt["path_cd"]["checkpoint"]

    def _load_checkpoint(self) -> None:
        load_path = self.opt.get("path_cd", {}).get("resume_state")
        if load_path is None:
            return
        gen_path = f"{load_path}_gen.pth"
        opt_path = f"{load_path}_opt.pth"
        logger.info("Loading CD model from %s", gen_path)
        self.net.load_state_dict(torch.load(gen_path, map_location=self.device))
        if self.phase == "train" and os.path.isfile(opt_path):
            state = torch.load(opt_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model and optimizer weights."""
        ckpt_dir = self._checkpoint_dir()
        os.makedirs(ckpt_dir, exist_ok=True)

        gen_path = os.path.join(ckpt_dir, f"cd_model_E{epoch}_gen.pth")
        opt_path = os.path.join(ckpt_dir, f"cd_model_E{epoch}_opt.pth")

        state_dict = {k: v.cpu() for k, v in self.net.state_dict().items()}
        torch.save(state_dict, gen_path)

        opt_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)

        if is_best:
            best_gen = os.path.join(ckpt_dir, "best_cd_model_gen.pth")
            best_opt = os.path.join(ckpt_dir, "best_cd_model_opt.pth")
            torch.save(state_dict, best_gen)
            torch.save(opt_state, best_opt)
            logger.info("Saved best CD model → %s", best_gen)
        logger.info("Saved CD model → %s", gen_path)

    # ------------------------------------------------------------------
    # Data feeding
    # ------------------------------------------------------------------

    def feed_data(
        self,
        feats_A: List[List[torch.Tensor]],
        feats_B: List[List[torch.Tensor]],
        data: dict,
    ) -> None:
        self.feats_A = feats_A
        self.feats_B = feats_B
        self.data = {
            k: v.to(self.device, dtype=torch.float) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def optimize_parameters(self) -> None:
        self.optimizer.zero_grad()
        self.pred_cm = self.net(self.feats_A, self.feats_B)
        loss = self.loss_func(self.pred_cm, self.data["L"].long())
        loss.backward()
        self.optimizer.step()
        self.log_dict["l_cd"] = loss.item()

    def test(self) -> None:
        self.net.eval()
        with torch.no_grad():
            self.pred_cm = self.net(self.feats_A, self.feats_B)
            loss = self.loss_func(self.pred_cm, self.data["L"].long())
            self.log_dict["l_cd"] = loss.item()
        self.net.train()

    def update_lr(self) -> None:
        self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # Metrics and logging
    # ------------------------------------------------------------------

    def get_current_log(self) -> dict:
        return self.log_dict

    def get_current_visuals(self) -> dict:
        return {
            "pred_cm": torch.argmax(self.pred_cm, dim=1, keepdim=False),
            "gt_cm": self.data["L"],
        }

    def _update_metric(self) -> float:
        pred = torch.argmax(self.pred_cm.detach(), dim=1)
        score = self.running_metric.update_cm(
            pr=pred.cpu().numpy(), gt=self.data["L"].detach().cpu().numpy()
        )
        return score

    def collect_running_batch_states(self) -> None:
        acc = self._update_metric()
        self.log_dict["running_acc"] = float(acc)

    def collect_epoch_states(self) -> None:
        scores = self.running_metric.get_scores()
        self.log_dict["epoch_acc"] = float(scores["mf1"])
        for k, v in scores.items():
            self.log_dict[k] = v

    def clear_cache(self) -> None:
        self.running_metric.clear()
