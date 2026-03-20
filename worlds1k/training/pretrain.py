"""Phase 1: train the hierarchical predictive model end-to-end.

Run directly::

    uv run python -m worlds1k.training.pretrain --dataset ucf101 --max-samples 8
    uv run python -m worlds1k.training.pretrain --dataset disney --max-samples 500 --num-epochs 50
    uv run python -m worlds1k.training.pretrain --list-datasets
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
class PretrainConfig:
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    eval_freq: int = 100
    eval_batches: int | None = None
    checkpoint_dir: Path | None = None
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class PretrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    frames_seen: list[int] = field(default_factory=list)


def _encode_batch(encoder: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(batch) == 1:
        return encoder(batch[0])
    return encoder(batch[0], batch[1])


class Pretrainer:
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
        config: PretrainConfig | None = None,
    ) -> None:
        self.config = config or PretrainConfig()
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        self.accelerator = Accelerator(mixed_precision="bf16", log_with="wandb" if use_wandb else None)
        self.global_step = 0
        self.frames_seen = 0

        all_params = list(model.parameters()) + list(encoder.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        self.model, self.encoder, self.optimizer, self.train_loader = self.accelerator.prepare(
            model, encoder, optimizer, train_loader
        )
        self.val_loader = self.accelerator.prepare(val_loader) if val_loader is not None else None

    def train(self, total_steps: int | None = None, run_name: str | None = None) -> PretrainResult:
        cfg = self.config
        self.accelerator.init_trackers(
            "worlds1k",
            config={"lr": cfg.learning_rate, "epochs": cfg.num_epochs, "warmup": cfg.warmup_steps},
            init_kwargs={"wandb": {"name": run_name}},
        )
        result = PretrainResult()
        t0 = time.monotonic()
        running_loss = 0.0
        running_n = 0

        # Estimate total steps for LR schedule if not provided.
        # Run one epoch to count steps, then use that.
        self._estimated_total = total_steps

        for _epoch in range(cfg.num_epochs):
            self.model.train()
            self.encoder.train()

            for batch in self.train_loader:
                # On first epoch, estimate total steps from actual batch count
                if self._estimated_total is None and self.global_step > 0 and _epoch == 0:
                    steps_per_epoch = self.global_step
                    self._estimated_total = steps_per_epoch * cfg.num_epochs

                lr = _cosine_lr(self.global_step, cfg.warmup_steps, cfg.learning_rate, self._estimated_total or 10000)
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
                    self.accelerator.print(
                        f"step {self.global_step} | {_ftime(time.monotonic() - t0)} | "
                        f"frames {_fmt(self.frames_seen)} | lr {lr:.2e} | train {tl:.4f} | val {vl:.4f}"
                    )
                    self.accelerator.log(
                        {"train_loss": tl, "val_loss": vl, "lr": lr, "frames": self.frames_seen},
                        step=self.global_step,
                    )
                    running_loss = 0.0
                    running_n = 0
                    if cfg.checkpoint_dir is not None:
                        self._save(cfg.checkpoint_dir)

                if total_steps and self.global_step >= total_steps:
                    break
            if total_steps and self.global_step >= total_steps:
                break

        if running_n > 0:
            tl = running_loss / running_n
            vl = self._evaluate()
            result.train_losses.append(tl)
            result.val_losses.append(vl)
            result.frames_seen.append(self.frames_seen)
            self.accelerator.print(
                f"final | {_ftime(time.monotonic() - t0)} | "
                f"frames {_fmt(self.frames_seen)} | train {tl:.4f} | val {vl:.4f}"
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
    p = argparse.ArgumentParser(prog="worlds1k.training.pretrain", description="Train the world model.")

    p.add_argument("--dataset", type=str, default=None, help="Registry name or HuggingFace path.")
    p.add_argument("--list-datasets", action="store_true", help="Print available datasets and exit.")
    p.add_argument("--max-samples", type=int, default=100, help="Clips to stream per epoch (default: 100).")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--cache-dir", type=str, default="auto", help="Tensor cache dir ('auto', path, or 'none').")

    p.add_argument("--num-levels", type=int, default=3)
    p.add_argument("--encoder", type=str, default="dinov2-small")
    p.add_argument("--checkpoint", type=Path, default=None, help="Resume from checkpoint.")

    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=None, help="Stop after N steps (overrides epochs).")
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

    from worlds1k.model.encoders import build_frame_encoder
    from worlds1k.model.frame_encoder import VideoEncoder
    from worlds1k.model.world_model import WorldModel, WorldModelConfig

    config = WorldModelConfig(num_levels=args.num_levels, backbone_name=args.encoder, image_size=args.image_size)
    model = WorldModel.from_config(config)
    encoder = VideoEncoder(build_frame_encoder(config))

    n_train = sum(p.numel() for p in list(model.parameters()) + list(encoder.parameters()) if p.requires_grad)
    n_frozen = sum(p.numel() for p in list(model.parameters()) + list(encoder.parameters()) if not p.requires_grad)
    print(f"encoder: {args.encoder} | params: {_fmt(n_train)} trainable, {_fmt(n_frozen)} frozen")  # noqa: T201

    from torch.utils.data import DataLoader

    from worlds1k.data import StreamingVideoDataset

    cache = None if args.cache_dir == "none" else (args.cache_dir if args.cache_dir != "auto" else "auto")
    train_ds = StreamingVideoDataset(
        args.dataset,
        max_samples=args.max_samples,
        window_size=args.window_size,
        image_size=args.image_size,
        split=args.split,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=cache,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)

    trainer_config = PretrainConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_freq=args.eval_freq,
        checkpoint_dir=args.output_dir,
    )
    trainer = Pretrainer(model, encoder, train_loader, config=trainer_config)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rc = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (args.output_dir / "run_config.json").write_text(json.dumps(rc, indent=2))

    result = trainer.train(total_steps=args.total_steps)

    if result.train_losses:
        print(f"done. final loss: {result.train_losses[-1]:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
