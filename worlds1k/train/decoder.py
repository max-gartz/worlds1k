"""Phase 2: train diffusion decoder with frozen encoder.

Uses HuggingFace Accelerate for mixed precision and device management.
Logs to wandb when WANDB_API_KEY is set.

Run directly::

    uv run python -m worlds1k.train.decoder \\
        --world-model checkpoints/latest.pt \\
        --model world-3L-small \\
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

    from worlds1k.model.diffusion_decoder import DiffusionDecoderBase

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
class DiffusionDecoderTrainConfig:
    max_frames: int = 100_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    eval_freq: int = 100
    grad_clip_norm: float = 1.0


@dataclass
class DiffusionDecoderTrainResult:
    train_losses: list[float] = field(default_factory=list)


class DiffusionDecoderTrainer:
    """Train diffusion decoder with frozen world model + encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: DiffusionDecoderBase,
        train_loader: DataLoader,
        config: DiffusionDecoderTrainConfig | None = None,
    ) -> None:
        self.config = config or DiffusionDecoderTrainConfig()
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        self.accelerator = Accelerator(mixed_precision="bf16", log_with="wandb" if use_wandb else None)

        # Frozen encoder stays on CPU — only the decoder goes on GPU
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        self.encoder = encoder.cpu()

        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        self.decoder, self.optimizer, self.train_loader = self.accelerator.prepare(
            decoder, optimizer, train_loader
        )

    def train(self) -> DiffusionDecoderTrainResult:
        cfg = self.config
        self.accelerator.init_trackers(
            "worlds1k-diffusion-decoder", config={"lr": cfg.learning_rate, "max_frames": cfg.max_frames}
        )
        result = DiffusionDecoderTrainResult()
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
                cpu_video = video.cpu().float()
                features = self.encoder(cpu_video) if len(batch) == 1 else self.encoder(cpu_video, batch[1].cpu())
                features = features.to(video.device)

            b, t = video.shape[:2]
            d = features.shape[-1]
            frames_seen += b * t

            # Flatten (frame, feature) pairs and process in small chunks
            frames_flat = video.reshape(b * t, *video.shape[2:])  # (B*T, C, H, W)
            feat_flat = features.reshape(b * t, d)  # (B*T, d_input)
            chunk_size = min(4, b * t)
            chunk_loss = 0.0
            n_chunks = 0

            for ci in range(0, b * t, chunk_size):
                f_chunk = frames_flat[ci : ci + chunk_size]
                feat_chunk = feat_flat[ci : ci + chunk_size]

                with self.accelerator.autocast():
                    out = self.decoder(f_chunk, feat_chunk)
                    loss = out["loss"] / max((b * t) // chunk_size, 1)

                self.accelerator.backward(loss)
                chunk_loss += out["loss"].item()
                n_chunks += 1

            if cfg.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.decoder.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += chunk_loss / n_chunks
            running_n += 1

            if running_n % cfg.eval_freq == 0:
                tl = running_loss / running_n
                result.train_losses.append(tl)
                pct = 100 * frames_seen / cfg.max_frames
                self.accelerator.print(
                    f"diffusion decoder | {_ftime(time.monotonic() - t0)} | "
                    f"frames {_fmt(frames_seen)}/{_fmt(cfg.max_frames)} ({pct:.0f}%) | loss {tl:.6f}"
                )
                self.accelerator.log({"diffusion_loss": tl, "frames": frames_seen}, step=frames_seen)
                running_loss = 0.0
                running_n = 0

        if running_n > 0:
            result.train_losses.append(running_loss / running_n)
        self.accelerator.end_training()
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="worlds1k.train.decoder", description="Train diffusion decoder (phase 2).")
    p.add_argument("--world-model", type=Path, default=None, help="Path to world model checkpoint (.pt) from phase 1.")
    p.add_argument("--model", type=str, default=None, help="Named world model config (e.g. 'world-3L-base').")
    p.add_argument("--dataset", type=str, default=None, help="Dataset name or HuggingFace path.")
    p.add_argument("--max-frames", type=int, default=100_000)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--decoder-arch", type=str, choices=["adagn", "unet"], default="adagn",
                    help="Decoder architecture: adagn (MPS) or unet (GPU cross-attention).")
    p.add_argument("--decoder-size", type=str, choices=["small", "base", "large"], default="base",
                    help="Decoder size tier.")
    p.add_argument("--list-decoders", action="store_true", help="Print decoder param counts and exit.")
    p.add_argument("--num-inference-steps", type=int, default=20, help="DDIM steps for sampling (default: 20).")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-freq", type=int, default=50)
    p.add_argument("--encoder", type=str, default="dinov2-small")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from worlds1k.model.configs import build_diffusion_decoder, get_config, list_decoder_configs

    if args.list_decoders:
        if args.model is None:
            print("error: --list-decoders requires --model (e.g. --model world-3L-base)")  # noqa: T201
            return
        world_cfg = get_config(args.model)
        print(f"  decoders for {args.model} (d_input={world_cfg.d_input}):")  # noqa: T201
        for name, count in sorted(list_decoder_configs(world_cfg).items()):
            print(f"    {name:<20s} {_fmt(count):>6s} params")  # noqa: T201
        return

    from worlds1k.data import StreamingVideoDataset

    if args.model is None:
        print("error: --model is required for training")  # noqa: T201
        return
    if args.dataset is None:
        print("error: --dataset is required for training")  # noqa: T201
        return
    if args.output_dir is None:
        print("error: --output-dir is required for training")  # noqa: T201
        return

    config = get_config(args.model)
    config.image_size = args.image_size

    from worlds1k.model.encoder_base import build_vision_encoder
    from worlds1k.model.vision_encoder import VideoEncoder

    encoder = VideoEncoder(build_vision_encoder(config))

    # Load encoder weights from world model checkpoint if provided
    if args.world_model is not None:
        ckpt = torch.load(args.world_model, map_location="cpu", weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])

    from torch.utils.data import DataLoader

    ds = StreamingVideoDataset(
        args.dataset,
        window_size=args.window_size,
        image_size=args.image_size,
        max_videos=args.max_videos,
        token=os.environ.get("HF_TOKEN"),
    )
    loader = DataLoader(ds, batch_size=args.batch_size)
    cfg = DiffusionDecoderTrainConfig(
        max_frames=args.max_frames, learning_rate=args.learning_rate, eval_freq=args.eval_freq
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"training {args.decoder_arch}-{args.decoder_size} diffusion decoder...")  # noqa: T201
    decoder = build_diffusion_decoder(
        config, arch=args.decoder_arch, size=args.decoder_size, num_inference_steps=args.num_inference_steps
    )
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"decoder params: {_fmt(n_params)}")  # noqa: T201
    fr = DiffusionDecoderTrainer(encoder, decoder, loader, config=cfg).train()
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "arch": args.decoder_arch,
            "size": args.decoder_size,
            "d_model": decoder.d_model,
            "num_inference_steps": args.num_inference_steps,
        },
        args.output_dir / "vision_decoder.pt",
    )
    print(f"diffusion decoder done. loss: {fr.train_losses[-1]:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()
