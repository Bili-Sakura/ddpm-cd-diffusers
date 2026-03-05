"""
DDPMCDPipeline — DiffusionPipeline for DDPM-based change detection inference.

This pipeline loads a pre-trained SR3 UNet and a change-detection head,
then runs inference on a pair of images (before / after) to produce a
change map.

Usage::

    from src.pipelines import DDPMCDPipeline

    pipe = DDPMCDPipeline.from_pretrained("path/to/saved_pipeline")
    change_map = pipe(image_A, image_B)
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm

from src.models.diffusion import extract_features, precompute_alpha_tables


class DDPMCDPipeline(DiffusionPipeline):
    """Inference pipeline for DDPM-based change detection.

    Components (stored on disk by ``save_pretrained`` / loaded by
    ``from_pretrained``):

    * ``unet`` — SR3-style UNet (:class:`~src.models.unet.UNet`).
    * ``scheduler`` — :class:`~diffusers.DDPMScheduler`.
    * ``cd_head`` — change-detection head (``nn.Module``).

    The pipeline can also be used **without** a cd_head (``cd_head=None``)
    for unconditional image generation / sampling.
    """

    def __init__(self, unet, scheduler, cd_head=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        # cd_head is optional and may not be a ModelMixin
        self.cd_head = cd_head

    @torch.no_grad()
    def __call__(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        timesteps: List[int] = (50, 100, 400),
        feat_type: str = "dec",
    ) -> torch.Tensor:
        """Run change-detection inference.

        Args:
            image_A: pre-change image ``(B, 3, H, W)`` in ``[-1, 1]``.
            image_B: post-change image ``(B, 3, H, W)`` in ``[-1, 1]``.
            timesteps: diffusion timesteps at which to extract features.
            feat_type: ``"enc"`` or ``"dec"`` — which features to use.

        Returns:
            Change-map logits ``(B, n_classes, H, W)``.
        """
        if self.cd_head is None:
            raise RuntimeError(
                "DDPMCDPipeline requires a cd_head for change detection. "
                "Use the unet/scheduler directly for image generation."
            )

        sqrt_alphas = precompute_alpha_tables(self.scheduler)

        feats_A, feats_B = [], []
        for t in timesteps:
            fe_A, fd_A = extract_features(self.unet, image_A, t, sqrt_alphas)
            fe_B, fd_B = extract_features(self.unet, image_B, t, sqrt_alphas)
            if feat_type == "dec":
                feats_A.append(fd_A)
                feats_B.append(fd_B)
            else:
                feats_A.append(fe_A)
                feats_B.append(fe_B)

        change_map = self.cd_head(feats_A, feats_B)
        return change_map

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        in_channels: int = 3,
        image_size: int = 256,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Unconditional image generation (sampling).

        Args:
            batch_size: number of images to generate.
            in_channels: number of image channels.
            image_size: spatial resolution.
            num_inference_steps: if ``None``, uses all training timesteps.
            generator: optional torch Generator for reproducibility.

        Returns:
            Generated images ``(B, C, H, W)`` in ``[-1, 1]``.
        """
        device = self.unet.device if hasattr(self.unet, 'device') else next(self.unet.parameters()).device
        num_timesteps = self.scheduler.config.num_train_timesteps
        sqrt_alphas = precompute_alpha_tables(self.scheduler)

        image = torch.randn(
            (batch_size, in_channels, image_size, image_size),
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps or num_timesteps)

        for i in tqdm(reversed(range(num_timesteps)), total=num_timesteps, desc="Sampling"):
            noise_level = torch.FloatTensor(
                [sqrt_alphas[i + 1]]
            ).repeat(batch_size, 1).to(device)

            noise_pred = self.unet(image, noise_level)

            # Use scheduler step
            self.scheduler.set_timesteps(num_timesteps)
            image = self.scheduler.step(noise_pred, i, image).prev_sample

        return image
