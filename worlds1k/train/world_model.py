"""Phase 1: train the hierarchical predictive model end-to-end.

Run directly::

    uv run python -m worlds1k.train.world_model --dataset ucf101 --max-frames 5000
    uv run python -m worlds1k.train.world_model --dataset disney --max-frames 500000
    uv run python -m worlds1k.train.world_model --list-datasets
"""

from __future__ import annotations

import argparse
import json
import math
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
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _cosine_lr(step: int, warmup: int, base_lr: float, total: int) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class WorldModelTrainConfig:
    max_frames: int = 100_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    eval_freq: int = 100
    eval_batches: int | None = None
    checkpoint_dir: Path | None = None
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class WorldModelTrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    frames_seen: list[int] = field(default_factory=list)


def _encode_batch(encoder: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(batch) == 1:
        return encoder(batch[0])
    return encoder(batch[0], batch[1])


class WorldModelTrainer:
    """Phase 1 training loop using HuggingFace Accelerate.

    Handles mixed precision, device placement, gradient scaling,
    and distributed training automatically.
    """

    def __init__(
        self,
        model: WorldModel,
        encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: WorldModelTrainConfig | None = None,
    ) -> None:
        self.config = config or WorldModelTrainConfig()
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        self.accelerator = Accelerator(mixed_precision="bf16", log_with="wandb" if use_wandb else None)
        self.global_step = 0
        self.frames_seen = 0

        all_params = list(model.parameters()) + list(encoder.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Store frames_per_step before accelerator wraps the loader
        self._frames_per_step = train_loader.batch_size * 128  # window_size default; updated on first batch

        self.model, self.encoder, self.optimizer, self.train_loader = self.accelerator.prepare(
            model, encoder, optimizer, train_loader
        )
        self.val_loader = self.accelerator.prepare(val_loader) if val_loader is not None else None

    def train(self, run_name: str | None = None) -> WorldModelTrainResult:
        cfg = self.config
        self.accelerator.init_trackers(
            "worlds1k-world-model",
            config={"lr": cfg.learning_rate, "max_frames": cfg.max_frames, "warmup": cfg.warmup_steps},
            init_kwargs={"wandb": {"name": run_name}},
        )
        result = WorldModelTrainResult()
        t0 = time.monotonic()
        running_loss = 0.0
        running_n = 0

        total_steps = cfg.max_frames // max(self._frames_per_step, 1)

        self.model.train()
        self.encoder.train()

        for batch in self.train_loader:
            if self.frames_seen >= cfg.max_frames:
                break

            # Update frames_per_step from actual batch on first step
            if self.global_step == 0:
                self._frames_per_step = batch[0].shape[0] * batch[0].shape[1]
                total_steps = cfg.max_frames // max(self._frames_per_step, 1)

            lr = _cosine_lr(self.global_step, cfg.warmup_steps, cfg.learning_rate, total_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with self.accelerator.autocast():
                features = _encode_batch(self.encoder, batch)
                loss = self.model(features)["loss"]

            self.accelerator.backward(loss)
            if cfg.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.frames_seen += batch[0].shape[0] * batch[0].shape[1]
            self.global_step += 1
            running_loss += loss.item()
            running_n += 1

            if self.global_step % cfg.eval_freq == 0:
                tl = running_loss / running_n
                vl = self._evaluate()
                result.train_losses.append(tl)
                result.val_losses.append(vl)
                result.frames_seen.append(self.frames_seen)
                pct = 100 * self.frames_seen / cfg.max_frames
                self.accelerator.print(
                    f"step {self.global_step} | {_ftime(time.monotonic() - t0)} | "
                    f"frames {_fmt(self.frames_seen)}/{_fmt(cfg.max_frames)} ({pct:.0f}%) | "
                    f"lr {lr:.2e} | train {tl:.4f}"
                )
                self.accelerator.log(
                    {"train_loss": tl, "val_loss": vl, "lr": lr, "frames": self.frames_seen},
                    step=self.global_step,
                )
                running_loss = 0.0
                running_n = 0
                if cfg.checkpoint_dir is not None:
                    self._save(cfg.checkpoint_dir)

        if running_n > 0:
            tl = running_loss / running_n
            result.train_losses.append(tl)
            result.frames_seen.append(self.frames_seen)
            pct = 100 * self.frames_seen / cfg.max_frames
            self.accelerator.print(
                f"final | {_ftime(time.monotonic() - t0)} | "
                f"frames {_fmt(self.frames_seen)}/{_fmt(cfg.max_frames)} ({pct:.0f}%) | train {tl:.4f}"
            )
            if cfg.checkpoint_dir is not None:
                self._save(cfg.checkpoint_dir)

        self.accelerator.end_training()
        return result

    def _evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        self.encoder.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                features = _encode_batch(self.encoder, batch)
                total += self.model(features)["loss"].item()
                n += 1
                if self.config.eval_batches is not None and n >= self.config.eval_batches:
                    break
        self.model.train()
        self.encoder.train()
        return total / n if n > 0 else float("nan")

    def _save(self, checkpoint_dir: Path, keep_last: int = 3) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        state = {
            "model": unwrapped_model.state_dict(),
            "encoder": unwrapped_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "frames_seen": self.frames_seen,
        }
        path = checkpoint_dir / f"step_{self.global_step}.pt"
        self.accelerator.save(state, path)
        self.accelerator.save(state, checkpoint_dir / "latest.pt")
        # Prune old checkpoints
        old = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        for f in old[:-keep_last]:
            f.unlink()

    def save_checkpoint(self, path: str | Path) -> None:
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        self.accelerator.save(
            {
                "model": unwrapped_model.state_dict(),
                "encoder": unwrapped_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "frames_seen": self.frames_seen,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model"])
        self.accelerator.unwrap_model(self.encoder).load_state_dict(ckpt["encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.frames_seen = ckpt.get("frames_seen", 0)
        self.accelerator.print(f"resumed from step {self.global_step} ({_fmt(self.frames_seen)} frames)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="worlds1k.train.world_model", description="Train the world model.")

    p.add_argument("--dataset", type=str, default=None, help="Registry name or HuggingFace path.")
    p.add_argument("--list-datasets", action="store_true", help="Print available datasets and exit.")
    p.add_argument("--max-frames", type=int, default=100_000, help="Total frames to train on (default: 100K).")
    p.add_argument("--max-videos", type=int, default=None, help="Max source videos to download/cache.")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--cache-dir", type=str, default="auto", help="Tensor cache dir ('auto', path, or 'none').")
    p.add_argument("--with-audio", action="store_true", help="Decode audio and train with AudioVideoEncoder.")

    p.add_argument("--num-levels", type=int, default=3)
    p.add_argument("--encoder", type=str, default="dinov2-small")
    p.add_argument("--checkpoint", type=Path, default=None, help="Resume from checkpoint.")

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-freq", type=int, default=100)

    p.add_argument("--output-dir", type=Path, default=None, help="Checkpoint directory.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_datasets:
        from worlds1k.data import list_datasets

        for name, spec in list_datasets().items():
            gated = " [gated]" if spec.gated else ""
            print(f"  {name:<20s} {spec.description}{gated}")  # noqa: T201
        return

    if args.dataset is None:
        print("error: --dataset is required (or use --list-datasets)")  # noqa: T201
        return

    from worlds1k.model.world_model import WorldModel, WorldModelConfig

    if args.with_audio:
        from worlds1k.model.audio_encoder import AudioVideoEncoder

        config = WorldModelConfig(
            num_levels=args.num_levels,
            backbone_name=args.encoder,
            image_size=args.image_size,
            d_input=512 + 256,  # visual + audio
        )
        model = WorldModel.from_config(config)
        encoder = AudioVideoEncoder.from_pretrained(args.encoder, 512, "whisper-tiny", 256)
        mode = f"{args.encoder} + whisper-tiny"
    else:
        from worlds1k.model.encoder_base import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder

        config = WorldModelConfig(num_levels=args.num_levels, backbone_name=args.encoder, image_size=args.image_size)
        model = WorldModel.from_config(config)
        encoder = VideoEncoder(build_frame_encoder(config))
        mode = args.encoder

    n_train = sum(p.numel() for p in list(model.parameters()) + list(encoder.parameters()) if p.requires_grad)
    n_frozen = sum(p.numel() for p in list(model.parameters()) + list(encoder.parameters()) if not p.requires_grad)
    print(f"encoder: {mode} | params: {_fmt(n_train)} trainable, {_fmt(n_frozen)} frozen")  # noqa: T201

    from torch.utils.data import DataLoader

    from worlds1k.data import StreamingVideoDataset

    cache = None if args.cache_dir == "none" else (args.cache_dir if args.cache_dir != "auto" else "auto")
    train_ds = StreamingVideoDataset(
        args.dataset,
        window_size=args.window_size,
        image_size=args.image_size,
        split=args.split,
        token=os.environ.get("HF_TOKEN"),
        with_audio=args.with_audio,
        max_videos=args.max_videos,
        cache_dir=cache,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)

    frames_per_step = args.batch_size * args.window_size
    total_steps = args.max_frames // frames_per_step
    print(  # noqa: T201
        f"training: {_fmt(args.max_frames)} frames = {total_steps} steps "
        f"({args.batch_size} x {args.window_size} frames/step)"
    )

    trainer_config = WorldModelTrainConfig(
        max_frames=args.max_frames,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_freq=args.eval_freq,
        checkpoint_dir=args.output_dir,
    )
    trainer = WorldModelTrainer(model, encoder, train_loader, config=trainer_config)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rc = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (args.output_dir / "run_config.json").write_text(json.dumps(rc, indent=2))

    result = trainer.train()

    if result.train_losses:
        print(f"done. final loss: {result.train_losses[-1]:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
