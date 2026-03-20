"""Phase 2: train frame and/or audio decoders with frozen encoder.

After the hierarchical predictive model is trained (phase 1), this module
trains decoders to map latent states back to observations. The world model
and encoder are frozen so latent representations remain stable.

- :class:`FrameDecoderTrainer` — latent → pixel frames (MSE loss)
- :class:`AudioDecoderTrainer` — latent → mel spectrograms (MSE loss)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
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


def _encode_batch(encoder: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor:
    """Encode a batch through the encoder (video-only or audio+video)."""
    if len(batch) == 1:
        return encoder(batch[0].to(device))
    return encoder(batch[0].to(device), batch[1].to(device))


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
        device = next(self.decoder.parameters()).device
        self.decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            video = batch[0].to(device)

            with torch.no_grad():
                features = _encode_batch(self.encoder, batch, device)
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
        device = next(self.audio_decoder.parameters()).device
        self.audio_decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            if len(batch) < 2:
                continue

            audio = batch[1].to(device)  # (B, T, n_mels, T_audio)

            with torch.no_grad():
                features = _encode_batch(self.encoder, batch, device)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="worlds1k.train.decoder", description="Train frame/audio decoders (phase 2).")
    p.add_argument("--checkpoint", type=Path, required=True, help="World model checkpoint from phase 1.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name or HuggingFace path.")
    p.add_argument("--max-frames", type=int, default=100_000)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-freq", type=int, default=50)
    p.add_argument("--with-audio", action="store_true", help="Also train audio decoder.")
    p.add_argument("--encoder", type=str, default="dinov2-small")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import os

    from worlds1k.data import StreamingVideoDataset
    from worlds1k.model.frame_decoder import FrameDecoder
    from worlds1k.model.world_model import WorldModel, WorldModelConfig

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")  # noqa: T201

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if args.with_audio:
        from worlds1k.model.audio_encoder import AudioVideoEncoder

        config = WorldModelConfig(image_size=args.image_size, d_input=512 + 256)
        model = WorldModel.from_config(config).to(device)
        model.load_state_dict(ckpt["model"])
        encoder = AudioVideoEncoder.from_pretrained(args.encoder, 512, "whisper-tiny", 256).to(device)
        encoder.load_state_dict(ckpt["encoder"])
    else:
        from worlds1k.model.encoder_base import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder

        config = WorldModelConfig(image_size=args.image_size, backbone_name=args.encoder)
        model = WorldModel.from_config(config).to(device)
        model.load_state_dict(ckpt["model"])
        encoder = VideoEncoder(build_frame_encoder(config)).to(device)
        encoder.load_state_dict(ckpt["encoder"])

    from torch.utils.data import DataLoader

    ds = StreamingVideoDataset(
        args.dataset,
        window_size=args.window_size,
        image_size=args.image_size,
        with_audio=args.with_audio,
        max_videos=args.max_videos,
        token=os.environ.get("HF_TOKEN"),
    )
    loader = DataLoader(ds, batch_size=args.batch_size)
    cfg = DecodeTrainConfig(max_frames=args.max_frames, learning_rate=args.learning_rate, eval_freq=args.eval_freq)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("training frame decoder...")  # noqa: T201
    frame_dec = FrameDecoder(config.d_latents[0], frame_height=args.image_size, frame_width=args.image_size).to(device)
    fr = FrameDecoderTrainer(model, encoder, frame_dec, loader, config=cfg).train()
    torch.save({"decoder": frame_dec.state_dict()}, args.output_dir / "frame_decoder.pt")
    print(f"frame decoder done. loss: {fr.train_losses[-1]:.6f}")  # noqa: T201

    if args.with_audio:
        from worlds1k.model.audio_decoder import AudioDecoder

        print("training audio decoder...")  # noqa: T201
        audio_dec = AudioDecoder(config.d_latents[0]).to(device)
        ar = AudioDecoderTrainer(model, encoder, audio_dec, loader, config=cfg).train()
        torch.save({"audio_decoder": audio_dec.state_dict()}, args.output_dir / "audio_decoder.pt")
        print(f"audio decoder done. loss: {ar.train_losses[-1]:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()
