"""Phase 2: train frame and/or audio decoders with frozen encoder.

After the hierarchical predictive model is trained (phase 1), this module
trains decoders to map latent states back to observations. The world model
and encoder are frozen so latent representations remain stable.

- :class:`FrameDecoderTrainer` — latent → pixel frames (MSE loss)
- :class:`AudioDecoderTrainer` — latent → mel spectrograms (MSE loss)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from worlds1k.model.world_model import WorldModel

log = logging.getLogger(__name__)


@dataclass
class DecodeTrainConfig:
    max_frames: int = 100_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    eval_freq: int = 100
    grad_clip_norm: float = 1.0


@dataclass
class DecodeTrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def _encode_batch(encoder: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Encode a batch through the encoder (video-only or audio+video)."""
    if len(batch) == 1:
        return encoder(batch[0])
    return encoder(batch[0], batch[1])


class FrameDecoderTrainer:
    """Phase 2: train frame decoder (latent → frames) with everything else frozen."""

    def __init__(
        self,
        world_model: WorldModel,
        encoder: nn.Module,
        decoder: nn.Module,
        train_loader: DataLoader,
        config: DecodeTrainConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.config = config or DecodeTrainConfig()

        for p in self.world_model.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.world_model.eval()
        self.encoder.eval()

        self.optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

    def train(self) -> DecodeTrainResult:
        cfg = self.config
        result = DecodeTrainResult()
        frames_seen = 0
        running_loss = 0.0
        running_n = 0
        self.decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            video = batch[0].to(next(self.decoder.parameters()).device)

            with torch.no_grad():
                features = _encode_batch(self.encoder, batch)
                z = self.world_model(features)["z"][0]  # level-0 latents

            b, t, d = z.shape
            frames_seen += b * t

            recon = self.decoder(z.reshape(b * t, d)).view(b, t, *video.shape[2:])
            loss = nn.functional.mse_loss(recon, video)

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()

            running_loss += loss.item()
            running_n += 1

            if running_n % cfg.eval_freq == 0:
                result.train_losses.append(running_loss / running_n)
                running_loss = 0.0
                running_n = 0

        if running_n > 0:
            result.train_losses.append(running_loss / running_n)
        return result


class AudioDecoderTrainer:
    """Phase 2: train audio decoder (latent → mel spectrograms) with everything else frozen.

    Expects the dataloader to yield ``(video, audio)`` tuples where audio is
    ``(B, T, n_mels, T_audio)`` — per-frame mel spectrograms.
    """

    def __init__(
        self,
        world_model: WorldModel,
        encoder: nn.Module,
        audio_decoder: nn.Module,
        train_loader: DataLoader,
        config: DecodeTrainConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.encoder = encoder
        self.audio_decoder = audio_decoder
        self.train_loader = train_loader
        self.config = config or DecodeTrainConfig()

        for p in self.world_model.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.world_model.eval()
        self.encoder.eval()

        self.optimizer = torch.optim.AdamW(
            audio_decoder.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

    def train(self) -> DecodeTrainResult:
        cfg = self.config
        result = DecodeTrainResult()
        frames_seen = 0
        running_loss = 0.0
        running_n = 0
        self.audio_decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            if len(batch) < 2:
                continue  # skip batches without audio

            video = batch[0].to(next(self.audio_decoder.parameters()).device)
            audio = batch[1].to(video.device)  # (B, T, n_mels, T_audio)

            with torch.no_grad():
                features = _encode_batch(self.encoder, batch)
                z = self.world_model(features)["z"][0]  # level-0 latents

            b, t, d = z.shape
            frames_seen += b * t

            # Decode each latent to a mel chunk
            mel_pred = self.audio_decoder(z.reshape(b * t, d))  # (B*T, n_mels, t_mel)
            mel_pred = mel_pred.view(b, t, mel_pred.shape[1], mel_pred.shape[2])

            # Target: truncate or pad ground truth audio to match decoder output size
            t_mel = mel_pred.shape[3]
            target = audio[:, :, :, :t_mel]
            if target.shape[3] < t_mel:
                target = nn.functional.pad(target, (0, t_mel - target.shape[3]))

            loss = nn.functional.mse_loss(mel_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.audio_decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()

            running_loss += loss.item()
            running_n += 1

            if running_n % cfg.eval_freq == 0:
                result.train_losses.append(running_loss / running_n)
                running_loss = 0.0
                running_n = 0

        if running_n > 0:
            result.train_losses.append(running_loss / running_n)
        return result
