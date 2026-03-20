"""Tests for the training pipeline."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from worlds1k.data import SyntheticVideoDataset
from worlds1k.model.world_model import WorldModel, WorldModelConfig


def _small_config(**overrides: object) -> WorldModelConfig:
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

        loss0 = model(features)["loss"].item()
        for _ in range(20):
            loss = model(features)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert model(features)["loss"].item() < loss0


class TestPretrainer:
    def test_train_video_only(self):
        from worlds1k.train.world_model import PretrainConfig, Pretrainer

        config = _small_config()
        model = WorldModel.from_config(config)

        class _FlatEncoder(torch.nn.Module):
            def __init__(self, d_out: int):
                super().__init__()
                self.proj = torch.nn.Linear(3 * 16 * 16, d_out)

            def forward(self, video: torch.Tensor) -> torch.Tensor:
                b, t = video.shape[:2]
                return self.proj(video.reshape(b, t, -1))

        ds = SyntheticVideoDataset(window_size=8, image_size=16)
        loader = DataLoader(ds, batch_size=2)
        encoder = _FlatEncoder(32)

        # 64 frames = 4 steps at batch_size=2, window_size=8
        train_cfg = PretrainConfig(max_frames=64, eval_freq=1, warmup_steps=0)
        trainer = Pretrainer(model, encoder, loader, config=train_cfg)
        result = trainer.train()

        assert len(result.train_losses) > 0

    def test_train_audio_video(self):
        from worlds1k.train.world_model import PretrainConfig, Pretrainer

        d_input = 48
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

        ds = SyntheticVideoDataset(window_size=8, image_size=16, include_audio=True, n_mels=80, audio_time_steps=100)
        loader = DataLoader(ds, batch_size=2)
        encoder = _MultimodalEncoder()

        train_cfg = PretrainConfig(max_frames=64, eval_freq=1, warmup_steps=0)
        trainer = Pretrainer(model, encoder, loader, config=train_cfg)
        result = trainer.train()

        assert len(result.train_losses) > 0

    def test_stops_at_max_frames(self):
        """Training stops even if max_frames is not a multiple of frames_per_step."""
        from worlds1k.train.world_model import PretrainConfig, Pretrainer

        config = _small_config()
        model = WorldModel.from_config(config)

        class _FlatEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3 * 16 * 16, 32)

            def forward(self, video: torch.Tensor) -> torch.Tensor:
                b, t = video.shape[:2]
                return self.proj(video.reshape(b, t, -1))

        ds = SyntheticVideoDataset(window_size=8, image_size=16)
        loader = DataLoader(ds, batch_size=2)

        # 50 frames with 16 frames/step = should stop after 3 steps (48 frames)
        train_cfg = PretrainConfig(max_frames=50, eval_freq=100, warmup_steps=0)
        trainer = Pretrainer(model, _FlatEncoder(), loader, config=train_cfg)
        trainer.train()

        assert trainer.frames_seen <= 50 + 16  # at most one extra batch


class TestCLI:
    def test_parse_dataset(self):
        from worlds1k.train.world_model import parse_args

        args = parse_args(["--dataset", "ucf101", "--max-frames", "10000"])
        assert args.dataset == "ucf101"
        assert args.max_frames == 10000

    def test_parse_defaults(self):
        from worlds1k.train.world_model import parse_args

        args = parse_args(["--dataset", "disney"])
        assert args.window_size == 128
        assert args.batch_size == 4
        assert args.image_size == 64
        assert args.max_frames == 100_000

    def test_list_datasets_flag(self):
        from worlds1k.train.world_model import parse_args

        args = parse_args(["--list-datasets"])
        assert args.list_datasets is True


@pytest.mark.integration
class TestFullPipeline:
    def test_end_to_end_with_streaming(self):
        from worlds1k.data import StreamingVideoDataset
        from worlds1k.model.encoder_base import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder
        from worlds1k.train.world_model import PretrainConfig, Pretrainer

        config = WorldModelConfig(num_levels=3, image_size=64)
        model = WorldModel.from_config(config)
        encoder = VideoEncoder(build_frame_encoder(config))

        ds = StreamingVideoDataset("ucf101", window_size=128, image_size=64, max_videos=2)
        loader = DataLoader(ds, batch_size=1)

        result = Pretrainer(
            model,
            encoder,
            loader,
            config=PretrainConfig(max_frames=256, eval_freq=1, warmup_steps=0),
        ).train()

        assert len(result.train_losses) > 0
