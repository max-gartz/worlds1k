"""Tests for the training pipeline — model creation, encoding, and training loop."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from worlds1k.data import SyntheticVideoDataset
from worlds1k.model.world_model import WorldModel, WorldModelConfig


def _small_config(**overrides: object) -> WorldModelConfig:
    """A minimal model config that runs fast on CPU."""
    defaults = {
        "num_levels": 2,
        "d_input": 32,
        "d_latents": [16, 8],
        "d_actions": [4, 2],
        "temporal_strides": [1, 4],
        "num_transformer_heads": 2,
        "num_transformer_layers": 1,
        "image_size": 16,
    }
    defaults.update(overrides)
    return WorldModelConfig(**defaults)


class TestWorldModelForward:
    def test_forward_returns_loss(self):
        config = _small_config()
        model = WorldModel.from_config(config)
        # window_size must be >= product of strides: 1*4 = 4, use 8 for 2 top-level latents
        features = torch.randn(2, 8, 32)
        outputs = model(features)
        assert "loss" in outputs
        assert outputs["loss"].shape == ()
        assert outputs["loss"].requires_grad

    def test_forward_level_losses(self):
        config = _small_config()
        model = WorldModel.from_config(config)
        features = torch.randn(1, 8, 32)
        outputs = model(features)
        assert outputs["level_losses"].shape == (2,)

    def test_loss_decreases_with_grad_step(self):
        config = _small_config()
        model = WorldModel.from_config(config)
        features = torch.randn(2, 8, 32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # First loss
        loss0 = model(features)["loss"].item()

        # Several gradient steps
        for _ in range(20):
            loss = model(features)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_final = model(features)["loss"].item()
        assert loss_final < loss0, f"Loss did not decrease: {loss0:.4f} -> {loss_final:.4f}"


class TestPretrainer:
    def test_train_video_only(self):
        from worlds1k.training.pretrain import PretrainConfig, Pretrainer

        config = _small_config()
        model = WorldModel.from_config(config)

        # Simple identity encoder (features = random projection of video frames)
        encoder = torch.nn.Linear(3 * 16 * 16, 32)

        ds = SyntheticVideoDataset(num_samples=4, window_size=8, image_size=16)
        loader = DataLoader(ds, batch_size=2)

        class _FlatEncoder(torch.nn.Module):
            """Flatten video frames and project to d_input."""

            def __init__(self, d_out: int):
                super().__init__()
                self.proj = torch.nn.Linear(3 * 16 * 16, d_out)

            def forward(self, video: torch.Tensor) -> torch.Tensor:
                b, t, c, h, w = video.shape
                flat = video.reshape(b, t, c * h * w)
                return self.proj(flat)

        encoder = _FlatEncoder(32)
        train_cfg = PretrainConfig(num_epochs=2, eval_freq=1, warmup_steps=0)
        trainer = Pretrainer(model, encoder, loader, config=train_cfg)
        result = trainer.train(total_steps=4)

        assert len(result.train_losses) > 0

    def test_train_audio_video(self):
        """Test that training works with audio+video batches."""
        from worlds1k.training.pretrain import PretrainConfig, Pretrainer

        d_input = 48  # visual(32) + audio(16)
        config = _small_config(d_input=d_input)
        model = WorldModel.from_config(config)

        class _MultimodalEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.v_proj = torch.nn.Linear(3 * 16 * 16, 32)
                self.a_proj = torch.nn.Linear(80 * 100, 16)

            def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
                b, t = video.shape[:2]
                v = self.v_proj(video.reshape(b, t, -1))
                a = self.a_proj(audio.reshape(b, t, -1))
                return torch.cat([v, a], dim=-1)

        ds = SyntheticVideoDataset(
            num_samples=4,
            window_size=8,
            image_size=16,
            include_audio=True,
            n_mels=80,
            audio_time_steps=100,
        )
        loader = DataLoader(ds, batch_size=2)
        encoder = _MultimodalEncoder()

        train_cfg = PretrainConfig(num_epochs=2, eval_freq=1, warmup_steps=0)
        trainer = Pretrainer(model, encoder, loader, config=train_cfg)
        result = trainer.train(total_steps=4)

        assert len(result.train_losses) > 0


class TestCLI:
    def test_parse_dataset(self):
        from worlds1k.training.pretrain import parse_args

        args = parse_args(["--dataset", "ucf101", "--max-samples", "10"])
        assert args.dataset == "ucf101"
        assert args.max_samples == 10

    def test_parse_defaults(self):
        from worlds1k.training.pretrain import parse_args

        args = parse_args(["--dataset", "disney"])
        assert args.window_size == 128
        assert args.batch_size == 4
        assert args.image_size == 64
        assert args.num_epochs == 100

    def test_list_datasets_flag(self):
        from worlds1k.training.pretrain import parse_args

        args = parse_args(["--list-datasets"])
        assert args.list_datasets is True


@pytest.mark.integration
class TestFullPipeline:
    def test_end_to_end_with_streaming(self):
        """Stream ucf101, build real DINOv2 encoder, train 1 epoch."""
        from worlds1k.data import StreamingVideoDataset
        from worlds1k.model.encoders import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder
        from worlds1k.training.pretrain import PretrainConfig, Pretrainer

        config = WorldModelConfig(num_levels=3, image_size=64)
        model = WorldModel.from_config(config)
        encoder = VideoEncoder(build_frame_encoder(config))

        ds = StreamingVideoDataset("ucf101", max_samples=2, window_size=128, image_size=64)
        loader = DataLoader(ds, batch_size=1)

        result = Pretrainer(
            model,
            encoder,
            loader,
            config=PretrainConfig(num_epochs=1, eval_freq=1),
        ).train()

        assert len(result.train_losses) > 0
