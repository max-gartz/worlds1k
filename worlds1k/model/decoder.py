"""Pixel decoder — maps level-1 latents back to pixel space.

This module is separate from the predictive hierarchy and is trained
in phase 2 with the encoder frozen.  It exists purely to visualize
and evaluate what the model has learned; the world model itself
operates entirely in latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """Decode level-1 latent states z^(1) back to pixel frames.

    Trained *after* the hierarchical predictive model (phase 2) with
    the level-1 encoder frozen.  Uses a standard deconvolutional
    architecture to map from latent space to image space.

    Architecture:
    1. Linear projection from d_latent to a spatial feature map.
    2. Sequence of ConvTranspose2d + BatchNorm + GELU blocks that
       progressively upsample to the target resolution.
    3. Final Conv2d + sigmoid for [0, 1] pixel output.

    For a 64x64 output, the upsampling path is::

        (B, 256, 4, 4) → (B, 128, 8, 8) → (B, 64, 16, 16)
        → (B, 32, 32, 32) → (B, C, 64, 64)

    Parameters
    ----------
    d_latent : int
        Dimensionality of the level-1 latent z^(1).
    frame_channels : int
        Number of output image channels (e.g. 3 for RGB).
    frame_height : int
        Output image height in pixels.
    frame_width : int
        Output image width in pixels.
    base_channels : int
        Number of channels at the first spatial layer (default 256).
    """

    def __init__(
        self,
        d_latent: int,
        *,
        frame_channels: int = 3,
        frame_height: int = 64,
        frame_width: int = 64,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.frame_channels = frame_channels
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.base_channels = base_channels

        # Initial spatial size (before upsampling)
        self.init_h = 4
        self.init_w = 4

        # Linear projection: d_latent → flat spatial features
        self.proj = nn.Linear(d_latent, base_channels * self.init_h * self.init_w)
        self.proj_norm = nn.BatchNorm1d(base_channels * self.init_h * self.init_w)

        # Determine the number of upsample stages needed
        # Each stage doubles spatial resolution: 4 → 8 → 16 → 32 → 64
        num_stages = 0
        size = self.init_h
        while size < frame_height:
            size *= 2
            num_stages += 1

        # Build upsampling blocks
        channels = [base_channels]
        for i in range(num_stages - 1):
            channels.append(max(base_channels // (2 ** (i + 1)), 32))
        channels.append(frame_channels)

        layers: list[nn.Module] = []
        for i in range(num_stages):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            is_last = i == num_stages - 1

            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.GELU())
            else:
                # Final layer: sigmoid for [0, 1] pixel range
                layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states to pixel frames.

        Parameters
        ----------
        z : torch.Tensor
            Level-1 latent states, shape ``(B, d_latent)``.

        Returns
        -------
        torch.Tensor
            Reconstructed frames, shape ``(B, C, H, W)``.
        """
        B = z.size(0)

        # Project and reshape to spatial feature map
        x = self.proj(z)  # (B, base_channels * init_h * init_w)
        x = self.proj_norm(x)
        x = x.view(B, self.base_channels, self.init_h, self.init_w)

        # Upsample through transposed convolutions
        return self.decoder(x)  # (B, C, H, W)
