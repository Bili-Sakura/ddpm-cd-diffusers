"""
DiffusionFeatureExtractor
=========================
Wraps a HuggingFace diffusers ``UNet2DModel`` together with a ``DDPMScheduler``
to extract multi-scale intermediate features for the change-detection head.

Feature extraction strategy
----------------------------
During inference, for each requested timestep *t*:

1. Gaussian noise is added to the input image at level *t* via the scheduler.
2. A forward pass is run through the frozen UNet.
3. The output of every ``up_block`` is captured through a forward hook.
4. The captured feature maps are returned in order from **shallowest**
   (largest spatial resolution, lowest channel count, index 0) to
   **deepest** (smallest spatial resolution, highest channel count, index N−1).

This ordering makes the list directly indexable by the ``feat_scales`` used
in :class:`~src.models.cd_head.ChangeDetectionHead`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers import DDPMScheduler, UNet2DModel

logger = logging.getLogger(__name__)


class DiffusionFeatureExtractor(nn.Module):
    """Wraps a pre-trained diffusers :class:`~diffusers.UNet2DModel` to extract
    intermediate up-block features.

    Parameters
    ----------
    unet:
        A :class:`~diffusers.UNet2DModel` instance (pre-trained or randomly
        initialised for the diffusion pre-training stage).
    scheduler:
        A :class:`~diffusers.DDPMScheduler` instance whose noise schedule is
        consistent with the UNet weights.
    """

    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        scheduler_subfolder: str = "scheduler",
        unet_subfolder: str = "unet",
        **kwargs,
    ) -> "DiffusionFeatureExtractor":
        """Load a pre-trained model saved with
        :meth:`save_pretrained`.

        The directory is expected to have the layout produced by
        :meth:`save_pretrained`::

            checkpoint_dir/
                unet/          ← UNet2DModel weights + config
                scheduler/     ← DDPMScheduler config

        Parameters
        ----------
        pretrained_model_name_or_path:
            Path to local directory or HuggingFace Hub model id.
        """
        import os

        base = pretrained_model_name_or_path
        unet_path = os.path.join(base, unet_subfolder)
        scheduler_path = os.path.join(base, scheduler_subfolder)

        unet = UNet2DModel.from_pretrained(
            unet_path if os.path.isdir(unet_path) else base, **kwargs
        )
        scheduler = DDPMScheduler.from_pretrained(
            scheduler_path if os.path.isdir(scheduler_path) else base
        )
        return cls(unet, scheduler)

    @classmethod
    def from_config(
        cls,
        unet_config: dict,
        scheduler_config: dict,
    ) -> "DiffusionFeatureExtractor":
        """Build from plain configuration dictionaries (as read from YAML).

        Parameters
        ----------
        unet_config:
            Keyword arguments forwarded to :class:`~diffusers.UNet2DModel`.
        scheduler_config:
            Keyword arguments forwarded to :class:`~diffusers.DDPMScheduler`.
        """
        unet = UNet2DModel(**unet_config)
        scheduler = DDPMScheduler(**scheduler_config)
        return cls(unet, scheduler)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """Save UNet weights and scheduler config to *save_directory*.

        The resulting layout is::

            save_directory/
                unet/
                scheduler/
        """
        import os

        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        logger.info("Saved DiffusionFeatureExtractor to %s", save_directory)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_out_channels(self) -> Tuple[int, ...]:
        """Channel counts for the **usable** feature levels, ordered shallowest → deepest.

        The last ``up_block`` in diffusers ``UNet2DModel`` does not perform any
        spatial upsampling (``add_upsample=False``), so it produces a feature
        map at the same spatial resolution as its predecessor.  We exclude it
        from the extracted features to ensure each level has a unique spatial
        resolution.

        The i-th up_block (0=deepest) outputs channels corresponding to
        ``block_out_channels[-1-i]``.  After skipping the last (shallowest,
        no-upsample) block and reversing so that index 0 = shallowest, the
        channel counts equal ``block_out_channels[1:]``.
        """
        # block_out_channels is [shallowest, ..., deepest].
        # up_block[0] (deepest) outputs block_out_channels[-1] channels.
        # up_block[N-1] (shallowest, no-upsample, skipped) outputs block_out_channels[0].
        # The usable levels correspond to block_out_channels[1:] in shallowest→deepest order.
        return tuple(self.unet.config.block_out_channels[1:])

    @property
    def num_up_blocks(self) -> int:
        """Number of usable up-blocks (excludes the final no-upsample block)."""
        return len(self.unet.up_blocks) - 1

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
        t: int,
    ) -> List[torch.Tensor]:
        """Add noise at timestep *t* and extract UNet up-block features.

        Parameters
        ----------
        x:
            Input image tensor of shape ``(B, C, H, W)`` normalised to
            ``[-1, 1]``.
        t:
            Integer diffusion timestep in ``[0, num_train_timesteps)``.

        Returns
        -------
        List[torch.Tensor]
            Feature maps from each ``up_block``, ordered **shallowest first**
            (index 0 = largest spatial size, lowest channels) to
            **deepest last** (index -1 = smallest spatial size, most channels).
            Length equals ``num_up_blocks``.
        """
        device = x.device
        noise = torch.randn_like(x)
        t_tensor = torch.full(
            (x.shape[0],), t, device=device, dtype=torch.long
        )
        x_noisy = self.scheduler.add_noise(x, noise, t_tensor)

        # Capture up_block outputs via forward hooks.
        # We skip the *last* up_block because it has ``add_upsample=False``
        # in diffusers UNet2DModel and therefore produces a feature map at the
        # same spatial resolution as its predecessor, creating duplicates.
        captured: List[torch.Tensor] = []
        usable_up_blocks = list(self.unet.up_blocks)[:-1]

        def _make_hook(idx: int):
            def _hook(
                module: nn.Module,
                inputs: tuple,
                output: Union[torch.Tensor, tuple],
            ) -> None:
                # Some diffusers versions return (hidden_states, ...) tuples;
                # we always want the hidden-state tensor.
                feat = output[0] if isinstance(output, tuple) else output
                captured.append(feat.detach())

            return _hook

        hooks = [
            up_block.register_forward_hook(_make_hook(i))
            for i, up_block in enumerate(usable_up_blocks)
        ]

        try:
            self.unet(x_noisy, t_tensor)
        finally:
            for h in hooks:
                h.remove()

        # up_blocks[0] is deepest (small spatial), up_blocks[-1] is shallowest.
        # Reverse so that index 0 corresponds to the shallowest level, making
        # feat_scales=[0, 1, …, N-1] intuitive.
        return list(reversed(captured))

    # ------------------------------------------------------------------
    # Forward (training the diffusion model itself)
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for a batch of noisy images.

        Used during diffusion pre-training (not during CD fine-tuning).

        Parameters
        ----------
        noisy_images:
            Noisy images tensor ``(B, C, H, W)``.
        timesteps:
            Integer timesteps tensor of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Predicted noise (or x_0 depending on ``prediction_type``),
            same shape as ``noisy_images``.
        """
        return self.unet(noisy_images, timesteps).sample
