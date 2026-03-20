"""Phase 2: train frame and/or audio decoders with frozen encoder.

Uses HuggingFace Accelerate for mixed precision and device management.
Logs to wandb when WANDB_API_KEY is set.

Run directly::

    uv run python -m worlds1k.train.decoder \\
        --checkpoint checkpoints/world_model.pt \\
        --dataset disney --max-frames 50000 \\
        --output-dir checkpoints/decoders
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from accelerate import Accelerator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from worlds1k.model.world_model import WorldModel

_SUFFIXES = ((1e9, "B"), (1e6, "M"), (1e3, "K"))


def _fmt(n: int) -> str:
    for t, s in _SUFFIXES:
        if n >= t:
            return f"{n / t:.1f}{s}"
    return str(n)


def _ftime(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s"


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


class FrameDecoderTrainer:
    """Phase 2: train frame decoder (latent -> frames) with Accelerate."""

    def __init__(
        self,
        world_model: WorldModel,
        encoder: nn.Module,
        decoder: nn.Module,
        train_loader: DataLoader,
        config: DecodeTrainConfig | None = None,
    ) -> None:
        self.config = config or DecodeTrainConfig()
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        self.accelerator = Accelerator(mixed_precision="bf16", log_with="wandb" if use_wandb else None)

        for p in world_model.parameters():
            p.requires_grad = False
        for p in encoder.parameters():
            p.requires_grad = False
        world_model.eval()
        encoder.eval()

        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        self.world_model, self.encoder, self.decoder, self.optimizer, self.train_loader = self.accelerator.prepare(
            world_model, encoder, decoder, optimizer, train_loader
        )

    def train(self) -> DecodeTrainResult:
        cfg = self.config
        self.accelerator.init_trackers(
            "worlds1k-decoder", config={"lr": cfg.learning_rate, "max_frames": cfg.max_frames}
        )
        result = DecodeTrainResult()
        t0 = time.monotonic()
        frames_seen = 0
        running_loss = 0.0
        running_n = 0
        self.decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break

            video = batch[0]
            with torch.no_grad():
                features = self.encoder(video) if len(batch) == 1 else self.encoder(video, batch[1])
                z = self.world_model(features)["z"][0]

            b, t, d = z.shape
            frames_seen += b * t

            with self.accelerator.autocast():
                recon = self.decoder(z.reshape(b * t, d)).view(b, t, *video.shape[2:])
                loss = nn.functional.mse_loss(recon, video)

            self.accelerator.backward(loss)
            if cfg.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            running_n += 1

            if running_n % cfg.eval_freq == 0:
                tl = running_loss / running_n
                result.train_losses.append(tl)
                pct = 100 * frames_seen / cfg.max_frames
                self.accelerator.print(
                    f"frame decoder | {_ftime(time.monotonic() - t0)} | "
                    f"frames {_fmt(frames_seen)}/{_fmt(cfg.max_frames)} ({pct:.0f}%) | loss {tl:.6f}"
                )
                self.accelerator.log({"frame_decoder_loss": tl, "frames": frames_seen}, step=frames_seen)
                running_loss = 0.0
                running_n = 0

        if running_n > 0:
            result.train_losses.append(running_loss / running_n)
        self.accelerator.end_training()
        return result


class AudioDecoderTrainer:
    """Phase 2: train audio decoder (latent -> mel) with Accelerate."""

    def __init__(
        self,
        world_model: WorldModel,
        encoder: nn.Module,
        audio_decoder: nn.Module,
        train_loader: DataLoader,
        config: DecodeTrainConfig | None = None,
    ) -> None:
        self.config = config or DecodeTrainConfig()
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        self.accelerator = Accelerator(mixed_precision="bf16", log_with="wandb" if use_wandb else None)

        for p in world_model.parameters():
            p.requires_grad = False
        for p in encoder.parameters():
            p.requires_grad = False
        world_model.eval()
        encoder.eval()

        optimizer = torch.optim.AdamW(
            audio_decoder.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        self.world_model, self.encoder, self.audio_decoder, self.optimizer, self.train_loader = (
            self.accelerator.prepare(world_model, encoder, audio_decoder, optimizer, train_loader)
        )

    def train(self) -> DecodeTrainResult:
        cfg = self.config
        self.accelerator.init_trackers(
            "worlds1k-audio-decoder", config={"lr": cfg.learning_rate, "max_frames": cfg.max_frames}
        )
        result = DecodeTrainResult()
        t0 = time.monotonic()
        frames_seen = 0
        running_loss = 0.0
        running_n = 0
        self.audio_decoder.train()

        for batch in self.train_loader:
            if frames_seen >= cfg.max_frames:
                break
            if len(batch) < 2:
                continue

            audio = batch[1]

            with torch.no_grad():
                features = self.encoder(batch[0], batch[1])
                z = self.world_model(features)["z"][0]

            b, t, d = z.shape
            frames_seen += b * t

            with self.accelerator.autocast():
                mel_pred = self.audio_decoder(z.reshape(b * t, d))
                mel_pred = mel_pred.view(b, t, mel_pred.shape[1], mel_pred.shape[2])
                t_mel = mel_pred.shape[3]
                target = audio[:, :, :, :t_mel]
                if target.shape[3] < t_mel:
                    target = nn.functional.pad(target, (0, t_mel - target.shape[3]))
                loss = nn.functional.mse_loss(mel_pred, target)

            self.accelerator.backward(loss)
            if cfg.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.audio_decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            running_n += 1

            if running_n % cfg.eval_freq == 0:
                tl = running_loss / running_n
                result.train_losses.append(tl)
                pct = 100 * frames_seen / cfg.max_frames
                self.accelerator.print(
                    f"audio decoder | {_ftime(time.monotonic() - t0)} | "
                    f"frames {_fmt(frames_seen)}/{_fmt(cfg.max_frames)} ({pct:.0f}%) | loss {tl:.6f}"
                )
                self.accelerator.log({"audio_decoder_loss": tl, "frames": frames_seen}, step=frames_seen)
                running_loss = 0.0
                running_n = 0

        if running_n > 0:
            result.train_losses.append(running_loss / running_n)
        self.accelerator.end_training()
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

    from worlds1k.data import StreamingVideoDataset
    from worlds1k.model.frame_decoder import FrameDecoder
    from worlds1k.model.world_model import WorldModel, WorldModelConfig

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if args.with_audio:
        from worlds1k.model.audio_encoder import AudioVideoEncoder

        config = WorldModelConfig(image_size=args.image_size, d_input=512 + 256)
        model = WorldModel.from_config(config)
        model.load_state_dict(ckpt["model"])
        encoder = AudioVideoEncoder.from_pretrained(args.encoder, 512, "whisper-tiny", 256)
        encoder.load_state_dict(ckpt["encoder"])
    else:
        from worlds1k.model.encoder_base import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder

        config = WorldModelConfig(image_size=args.image_size, backbone_name=args.encoder)
        model = WorldModel.from_config(config)
        model.load_state_dict(ckpt["model"])
        encoder = VideoEncoder(build_frame_encoder(config))
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
    frame_dec = FrameDecoder(config.d_latents[0], frame_height=args.image_size, frame_width=args.image_size)
    fr = FrameDecoderTrainer(model, encoder, frame_dec, loader, config=cfg).train()
    torch.save({"decoder": frame_dec.state_dict()}, args.output_dir / "frame_decoder.pt")
    print(f"frame decoder done. loss: {fr.train_losses[-1]:.6f}")  # noqa: T201

    if args.with_audio:
        from worlds1k.model.audio_decoder import AudioDecoder

        print("training audio decoder...")  # noqa: T201
        audio_dec = AudioDecoder(config.d_latents[0])
        ar = AudioDecoderTrainer(model, encoder, audio_dec, loader, config=cfg).train()
        torch.save({"audio_decoder": audio_dec.state_dict()}, args.output_dir / "audio_decoder.pt")
        print(f"audio decoder done. loss: {ar.train_losses[-1]:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()
