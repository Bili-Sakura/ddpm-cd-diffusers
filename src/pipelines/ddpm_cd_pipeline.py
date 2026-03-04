"""
DDPMCDPipeline
==============
End-to-end pipeline that combines a frozen
:class:`~src.models.diffusion_extractor.DiffusionFeatureExtractor` with a
:class:`~src.models.cd_head.ChangeDetectionHead` for change-detection
inference.

Typical usage::

    from src.pipelines import DDPMCDPipeline

    pipe = DDPMCDPipeline.from_pretrained(
        diffusion_ckpt="experiments/ddpm-pretrained",
        cd_ckpt="experiments/cd-finetuned/best_cd_model",
        feat_scales=[0, 1, 2, 3, 4],
        block_out_channels=(128, 256, 512, 1024, 1024),
        time_steps=[50, 100, 400],
    )
    change_map = pipe(image_A, image_B)  # (B, H, W) long tensor
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.diffusion_extractor import DiffusionFeatureExtractor
from ..models.cd_head import ChangeDetectionHead

logger = logging.getLogger(__name__)


class DDPMCDPipeline(nn.Module):
    """Inference pipeline: diffusion feature extraction → change detection.

    The diffusion model weights are kept **frozen** during CD inference.

    Parameters
    ----------
    extractor:
        Pre-trained :class:`~src.models.diffusion_extractor.DiffusionFeatureExtractor`.
    cd_head:
        Trained :class:`~src.models.cd_head.ChangeDetectionHead`.
    time_steps:
        Diffusion timesteps used for feature extraction.
    feat_scales:
        Up-block indices to feed into the CD head (0 = shallowest).
    feat_type:
        Currently only ``"dec"`` (decoder/up-block features) is supported.
    """

    def __init__(
        self,
        extractor: DiffusionFeatureExtractor,
        cd_head: ChangeDetectionHead,
        time_steps: List[int],
        feat_scales: List[int],
        feat_type: str = "dec",
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.cd_head = cd_head
        self.time_steps = time_steps
        self.feat_scales = feat_scales
        self.feat_type = feat_type

        # Freeze the diffusion model
        for p in self.extractor.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        diffusion_ckpt: str,
        cd_gen_path: str,
        feat_scales: List[int],
        block_out_channels: Sequence[int],
        out_channels: int = 2,
        img_size: int = 256,
        time_steps: Optional[List[int]] = None,
        feat_type: str = "dec",
        device: Optional[str] = None,
    ) -> "DDPMCDPipeline":
        """Load extractor and CD head from checkpoints.

        Parameters
        ----------
        diffusion_ckpt:
            Directory produced by
            :meth:`~src.models.diffusion_extractor.DiffusionFeatureExtractor.save_pretrained`.
        cd_gen_path:
            Path prefix for the CD model weights, e.g.
            ``"experiments/run/checkpoint/best_cd_model"``
            (the loader appends ``"_gen.pth"``).
        feat_scales:
            Up-block indices (0 = shallowest, N-1 = deepest).
        block_out_channels:
            Channel counts ordered shallowest → deepest, as returned by
            ``DiffusionFeatureExtractor.block_out_channels`` (i.e.
            ``unet_config["block_out_channels"][1:]``).
        """
        if time_steps is None:
            time_steps = [50, 100, 400]

        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        extractor = DiffusionFeatureExtractor.from_pretrained(diffusion_ckpt)
        extractor = extractor.to(_device)

        head = ChangeDetectionHead(
            feat_scales=feat_scales,
            block_out_channels=block_out_channels,
            out_channels=out_channels,
            img_size=img_size,
            time_steps=time_steps,
        ).to(_device)

        gen_path = f"{cd_gen_path}_gen.pth"
        head.load_state_dict(torch.load(gen_path, map_location=_device))
        logger.info("Loaded CD head from %s", gen_path)

        return cls(extractor, head, time_steps, feat_scales, feat_type)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.extractor.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> torch.Tensor:
        """Run change detection on a pair of images.

        Parameters
        ----------
        image_A, image_B:
            Image tensors of shape ``(B, C, H, W)`` normalised to ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Predicted change map of shape ``(B, H, W)`` with integer class labels.
        """
        self.extractor.eval()
        self.cd_head.eval()

        feats_A: List[List[torch.Tensor]] = []
        feats_B: List[List[torch.Tensor]] = []

        for t in self.time_steps:
            feats_A.append(self.extractor.extract_features(image_A, t))
            feats_B.append(self.extractor.extract_features(image_B, t))

        logits = self.cd_head(feats_A, feats_B)
        return torch.argmax(logits, dim=1)
