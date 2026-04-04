"""Diffusion decoder — renders frames from world model latents.

Two U-Net backends are available:

- **AdaGNDiffusionDecoder** — lightweight custom U-Net with Adaptive Group
  Normalisation.  No attention layers; runs on MPS / consumer GPUs.
- **CrossAttnDiffusionDecoder** — ``diffusers.UNet2DConditionModel`` with
  cross-attention conditioning.  More expressive but needs a CUDA GPU.

Both inherit from ``DiffusionDecoderBase`` and share the same public API
(``forward``, ``sample``) so the trainer and inference code work identically.

Training uses noise prediction (epsilon parameterisation) with a DDPM
schedule.  Inference uses DDIM sampling for fast generation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from diffusers import DDIMScheduler, DDPMScheduler
from torch import Tensor

# ===================================================================
# Shared building blocks
# ===================================================================


class SinusoidalTimestepEmbedding(nn.Module):
    """Map integer timesteps to sinusoidal positional embeddings."""

    def __init__(self, d_emb: int) -> None:
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t: Tensor) -> Tensor:
        half = self.d_emb // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


def _compute_n_stages(image_size: int) -> int:
    n, s = 0, image_size
    while s > 4 and n < 4:
        s = (s + 1) // 2
        n += 1
    return max(n, 2)


# ===================================================================
# Base class
# ===================================================================


class DiffusionDecoderBase(ABC, nn.Module):
    """Abstract base for diffusion decoders.

    Subclasses implement ``_predict_noise`` — everything else (noise
    scheduling, training loss, DDIM sampling) is handled here.
    """

    def __init__(
        self,
        d_latent: int,
        image_size: int,
        image_channels: int,
        d_model: int,
        arch: str,
        num_train_timesteps: int,
        num_inference_steps: int,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.image_size = image_size
        self.image_channels = image_channels
        self.d_model = d_model
        self.arch = arch
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule="linear", prediction_type="epsilon"
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule="linear", prediction_type="epsilon"
        )

    @abstractmethod
    def _predict_noise(self, noisy: Tensor, z: Tensor, timesteps: Tensor) -> Tensor:
        """Predict noise given noisy image, conditioning latent, and timesteps."""

    def forward(self, x: Tensor, z: Tensor) -> dict[str, Tensor]:
        """Training forward: predict noise.

        Parameters
        ----------
        x : Tensor
            Clean frames ``(B, C, H, W)`` in ``[0, 1]``.
        z : Tensor
            Conditioning latents ``(B, d_latent)``.

        Returns
        -------
        dict[str, Tensor]
            ``{"loss": mse}`` — noise prediction MSE.
        """
        B = x.size(0)
        device = x.device
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        noisy = self.train_scheduler.add_noise(x, noise, timesteps)
        pred = self._predict_noise(noisy, z, timesteps)
        return {"loss": nn.functional.mse_loss(pred, noise)}

    @torch.no_grad()
    def sample(self, z: Tensor) -> Tensor:
        """Generate frames from latents using DDIM sampling.

        Parameters
        ----------
        z : Tensor
            Conditioning latents ``(B, d_latent)``.

        Returns
        -------
        Tensor
            Generated images ``(B, C, H, W)`` in ``[0, 1]``.
        """
        B = z.size(0)
        device = z.device
        sample = torch.randn(B, self.image_channels, self.image_size, self.image_size, device=device)
        self.inference_scheduler.set_timesteps(self.num_inference_steps, device=device)
        for t in self.inference_scheduler.timesteps:
            pred = self._predict_noise(sample, z, t.expand(B))
            sample = self.inference_scheduler.step(pred, t, sample).prev_sample
        return sample.clamp(0.0, 1.0)


# ===================================================================
# AdaGN backend (MPS-friendly, no attention)
# ===================================================================


class _AdaGNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, d_cond: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.ada1 = nn.Linear(d_cond, out_ch * 2)
        self.ada2 = nn.Linear(d_cond, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # AdaGN after conv1 (operates on out_ch)
        scale1, shift1 = self.ada1(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        h = self.act(h)
        # AdaGN after conv2 — ada2 is available but we apply it as residual modulation
        scale2, shift2 = self.ada2(cond).chunk(2, dim=-1)
        h = self.conv2(h) * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        return h + self.skip(x)


class _Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class _AdaGNUNet(nn.Module):
    def __init__(
        self, image_channels: int, d_model: int, d_cond: int, n_stages: int, blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        channels = [d_model * (2**i) for i in range(n_stages)]
        self.input_conv = nn.Conv2d(image_channels, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = channels[0]
        for i in range(n_stages):
            ch = channels[i]
            self.down_blocks.append(nn.ModuleList([
                _AdaGNResBlock(prev_ch if j == 0 else ch, ch, d_cond) for j in range(blocks_per_stage)
            ]))
            prev_ch = ch
            self.down_samples.append(_Downsample(ch) if i < n_stages - 1 else nn.Identity())

        mid = channels[-1]
        self.mid1 = _AdaGNResBlock(mid, mid, d_cond)
        self.mid2 = _AdaGNResBlock(mid, mid, d_cond)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in reversed(range(n_stages)):
            ch_out = channels[i]
            ch_in = channels[min(i + 1, n_stages - 1)] if i < n_stages - 1 else mid
            self.up_blocks.append(nn.ModuleList([
                _AdaGNResBlock(ch_in + ch_out if j == 0 else ch_out, ch_out, d_cond) for j in range(blocks_per_stage)
            ]))
            self.up_samples.append(_Upsample(ch_out) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], image_channels, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.input_conv(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.down_samples, strict=True):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)
        for blocks, up in zip(self.up_blocks, self.up_samples, strict=True):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, cond)
            h = up(h)
        return self.out_conv(self.out_act(self.out_norm(h)))


class AdaGNDiffusionDecoder(DiffusionDecoderBase):
    """Diffusion decoder with AdaGN conditioning (no attention, MPS-friendly).

    Parameters
    ----------
    d_latent, image_size, image_channels, d_model, num_train_timesteps,
    num_inference_steps : see ``DiffusionDecoderBase``.
    """

    def __init__(
        self,
        d_latent: int,
        image_size: int = 64,
        image_channels: int = 3,
        d_model: int = 64,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__(
            d_latent, image_size, image_channels, d_model, "adagn", num_train_timesteps, num_inference_steps,
        )
        d_cond = d_model * 4
        self.time_emb = SinusoidalTimestepEmbedding(d_cond)
        self.time_proj = nn.Sequential(nn.Linear(d_cond, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))
        self.latent_proj = nn.Sequential(nn.Linear(d_latent, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))
        self.unet = _AdaGNUNet(image_channels, d_model, d_cond, _compute_n_stages(image_size))

    def _predict_noise(self, noisy: Tensor, z: Tensor, timesteps: Tensor) -> Tensor:
        cond = self.time_proj(self.time_emb(timesteps)) + self.latent_proj(z)
        return self.unet(noisy, cond)


# ===================================================================
# Cross-attention backend (GPU, uses diffusers)
# ===================================================================


class CrossAttnDiffusionDecoder(DiffusionDecoderBase):
    """Diffusion decoder with cross-attention conditioning (needs CUDA GPU).

    Uses ``diffusers.UNet2DConditionModel`` internally.

    Parameters
    ----------
    d_latent, image_size, image_channels, d_model, num_train_timesteps,
    num_inference_steps : see ``DiffusionDecoderBase``.
    """

    def __init__(
        self,
        d_latent: int,
        image_size: int = 64,
        image_channels: int = 3,
        d_model: int = 128,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__(
            d_latent, image_size, image_channels, d_model, "unet", num_train_timesteps, num_inference_steps,
        )
        from diffusers import UNet2DConditionModel

        d_cross = d_model
        self._num_tokens = 4
        self._d_cross = d_cross
        self.latent_proj = nn.Linear(d_latent, self._num_tokens * d_cross)

        n_stages = _compute_n_stages(image_size)
        block_out = []
        ch = d_model
        for _ in range(n_stages):
            block_out.append(ch)
            ch = min(ch * 2, d_model * 2)

        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=image_channels,
            out_channels=image_channels,
            block_out_channels=tuple(block_out),
            cross_attention_dim=d_cross,
            layers_per_block=2,
            down_block_types=tuple(["CrossAttnDownBlock2D"] * (n_stages - 1) + ["DownBlock2D"]),
            up_block_types=tuple(["UpBlock2D"] + ["CrossAttnUpBlock2D"] * (n_stages - 1)),
            norm_num_groups=min(32, block_out[0]),
        )

    def _predict_noise(self, noisy: Tensor, z: Tensor, timesteps: Tensor) -> Tensor:
        enc = self.latent_proj(z).view(z.size(0), self._num_tokens, self._d_cross)
        return self.unet(noisy, timesteps, encoder_hidden_states=enc).sample

