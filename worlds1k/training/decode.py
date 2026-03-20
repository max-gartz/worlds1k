"""Phase 2: train the pixel decoder with frozen encoder.

After the hierarchical predictive model is trained (phase 1), this module
trains a :class:`~worlds1k.model.decoder.PixelDecoder` to map
level-1 latents z^(1) back to pixel space.  The world model's encoder
is frozen so latent representations remain stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from worlds1k.model.decoder import PixelDecoder
    from worlds1k.model.world_model import WorldModel

log = logging.getLogger(__name__)


@dataclass
class DecodeTrainConfig:
    """Hyperparameters for phase 2 decoder training."""

    max_frames: int = 100_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    eval_freq: int = 100
    checkpoint_dir: Path | None = None
    grad_clip_norm: float = 1.0


@dataclass
class DecodeTrainResult:
    """Metrics collected during decoder training."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class DecoderTrainer:
    """Handles phase 2 training: pixel decoder with frozen encoder.

    The world model and visual encoder are frozen.  Only the pixel decoder
    receives gradients.  For each video batch the trainer:

    1. Encodes frames through the frozen encoder → features.
    2. Runs features through the frozen world model → level-1 latents z.
    3. Decodes each latent back to pixel space via the decoder.
    4. Computes MSE loss against the original frames.

    Parameters
    ----------
    world_model : WorldModel
        Trained world model (will be frozen).
    encoder : nn.Module
        Visual encoder (will be frozen).
    decoder : PixelDecoder
        The pixel decoder to train.
    train_loader : DataLoader
        Training data loader yielding ``(video,)`` batches.
    val_loader : DataLoader or None
        Validation data loader (optional).
    config : DecodeTrainConfig or None
        Training hyperparameters.
    """

    def __init__(
        self,
        world_model: WorldModel,
        encoder: nn.Module,
        decoder: PixelDecoder,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: DecodeTrainConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or DecodeTrainConfig()

        # Freeze world model and encoder
        for param in self.world_model.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.world_model.eval()
        self.encoder.eval()

        self.optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _encode_to_latents(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video → features → level-1 latents. Returns (latents, target_frames)."""
        with torch.no_grad():
            features = self.encoder(video)  # (B, T, d_input)
            outputs = self.world_model(features)
            z_level0 = outputs["z"][0]  # (B, T, d_latent) — level-1 latents

        # Target: original video frames for reconstruction
        # z_level0 has one latent per frame (stride=1), so shapes align
        return z_level0, video

    def train(self) -> DecodeTrainResult:
        """Run the decoder training loop and return collected metrics."""
        cfg = self.config
        result = DecodeTrainResult()
        global_step = 0
        frames_seen = 0
        running_loss = 0.0
        running_count = 0

        self.decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            (video,) = batch
            video = video.to(self.decoder.proj.weight.device)

            z_latents, target = self._encode_to_latents(video)
            b, t, d = z_latents.shape
            frames_seen += b * t

            z_flat = z_latents.reshape(b * t, d)
            recon = self.decoder(z_flat).view(b, t, *video.shape[2:])
            loss = nn.functional.mse_loss(recon, target)

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()

            global_step += 1  # noqa: SIM113
            running_loss += loss.item()
            running_count += 1

            if global_step % cfg.eval_freq == 0:
                train_loss = running_loss / running_count
                val_loss = self._evaluate()
                result.train_losses.append(train_loss)
                result.val_losses.append(val_loss)
                running_loss = 0.0
                running_count = 0

        if running_count > 0:
            train_loss = running_loss / running_count
            result.train_losses.append(train_loss)
            log.info("decoder final | train %.6f", train_loss)

        return result

    def _evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")

        self.decoder.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                (video,) = batch
                video = video.to(self.decoder.proj.weight.device)
                z_latents, target = self._encode_to_latents(video)
                b, t, d = z_latents.shape
                recon = self.decoder(z_latents.reshape(b * t, d)).view(b, t, *target.shape[2:])
                total_loss += nn.functional.mse_loss(recon, target).item()
                count += 1

        self.decoder.train()
        return total_loss / count if count > 0 else float("nan")

    def save_checkpoint(self, path: str | Path) -> None:
        """Save decoder state dict to *path*."""
        path = Path(path)
        torch.save({"decoder": self.decoder.state_dict()}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load decoder state dict from *path*."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(checkpoint["decoder"])
