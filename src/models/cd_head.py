"""
ChangeDetectionHead
===================
Multi-scale change-detection head that fuses features extracted from a frozen
diffusion model at multiple timesteps.

Architecture (adapted from the original ``cd_head_v2``)
---------------------------------------------------------
Features are received as::

    feats_A[t_idx][scale_idx]  →  Tensor (B, C, H, W)

where *scale_idx* indexes the output of the ``i``-th ``up_block`` of the
diffuser UNet **after reversing** (0 = shallowest / largest spatial, N-1 =
deepest / smallest spatial).  This is exactly the order produced by
:meth:`~src.models.diffusion_extractor.DiffusionFeatureExtractor.extract_features`.

The ``feat_scales`` parameter selects which resolution levels to use.
Using all levels, e.g. ``feat_scales = [0, 1, 2, 3, 4]``, exploits every
up-block output of a 5-level UNet.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_in_channels(
    feat_scales: Sequence[int],
    block_out_channels: Sequence[int],
) -> int:
    """Return total input channels for a given set of feature scales.

    Parameters
    ----------
    feat_scales:
        Indices into the reversed up-block list (0 = shallowest).
    block_out_channels:
        Channel counts ordered **shallowest → deepest**, i.e. as stored in
        ``UNet2DModel.config.block_out_channels``.  Index 0 gives the channel
        count at the shallowest resolution level.
    """
    return sum(block_out_channels[s] for s in feat_scales)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ChannelSELayer(nn.Module):
    """Channel Squeeze-and-Excitation block."""

    def __init__(self, num_channels: int, reduction_ratio: int = 2) -> None:
        super().__init__()
        reduced = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = x.view(B, C, -1).mean(dim=2)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s))))
        return x * s.view(B, C, 1, 1)


class SpatialSELayer(nn.Module):
    """Spatial Squeeze-and-Excitation block."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.sigmoid(self.conv(x))
        return x * s


class ChannelSpatialSELayer(nn.Module):
    """Concurrent Channel + Spatial SE block."""

    def __init__(self, num_channels: int, reduction_ratio: int = 2) -> None:
        super().__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cSE(x) + self.sSE(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_steps: Sequence[int]) -> None:
        super().__init__()
        n = len(time_steps)
        self.block = nn.Sequential(
            nn.Conv2d(dim * n, dim, 1) if n > 1 else nn.Identity(),
            nn.ReLU() if n > 1 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Change-detection head
# ---------------------------------------------------------------------------


class ChangeDetectionHead(nn.Module):
    """Multi-scale change-detection head (v2) adapted for diffusers features.

    Parameters
    ----------
    feat_scales:
        Indices of the up-block outputs to use (0 = shallowest).
    block_out_channels:
        Channel counts of the diffuser UNet ordered **shallowest → deepest**,
        as returned by
        ``DiffusionFeatureExtractor.block_out_channels``.
    out_channels:
        Number of output classes (2 for binary change detection).
    img_size:
        Spatial size of the output change map.
    time_steps:
        List of diffusion timesteps at which features are extracted.
    """

    def __init__(
        self,
        feat_scales: List[int],
        block_out_channels: Sequence[int],
        out_channels: int = 2,
        img_size: int = 256,
        time_steps: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if time_steps is None:
            time_steps = [50]

        # Process from deepest → shallowest
        feat_scales = sorted(feat_scales, reverse=True)
        self.feat_scales = feat_scales
        self.block_out_channels = list(block_out_channels)
        self.img_size = img_size
        self.time_steps = time_steps

        self.decoder = nn.ModuleList()
        for i, scale in enumerate(self.feat_scales):
            dim = get_in_channels([scale], self.block_out_channels)
            self.decoder.append(Block(dim=dim, dim_out=dim, time_steps=time_steps))

            if i < len(self.feat_scales) - 1:
                dim_out = get_in_channels(
                    [self.feat_scales[i + 1]], self.block_out_channels
                )
                self.decoder.append(AttentionBlock(dim=dim, dim_out=dim_out))

        # The classifier input is `x` = upsampled output of the second-to-last
        # AttentionBlock, which has channel count = block_out_channels[feat_scales[-1]].
        dim_out = get_in_channels([self.feat_scales[-1]], self.block_out_channels)
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(
        self,
        feats_A: List[List[torch.Tensor]],
        feats_B: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute change map.

        Parameters
        ----------
        feats_A, feats_B:
            Each is a list of length ``len(time_steps)``.  Each element is
            a list of feature tensors indexed by scale (0 = shallowest).

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, out_channels, H, W)``.
        """
        x: Optional[torch.Tensor] = None
        lvl = 0

        for layer in self.decoder:
            if isinstance(layer, Block):
                scale = self.feat_scales[lvl]

                # Gather features across all timesteps, concatenating on channel dim
                f_A = feats_A[0][scale]
                f_B = feats_B[0][scale]
                for t_i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[t_i][scale]), dim=1)
                    f_B = torch.cat((f_B, feats_B[t_i][scale]), dim=1)

                diff = torch.abs(layer(f_A) - layer(f_B))
                if lvl != 0 and x is not None:
                    diff = diff + x
                lvl += 1
            else:
                diff = layer(diff)
                x = F.interpolate(diff, scale_factor=2, mode="bilinear", align_corners=False)

        # Classify using the last upsampled feature map
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))
        return cm
