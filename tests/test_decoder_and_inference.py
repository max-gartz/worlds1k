"""Tests for decoder training (phase 2) and inference (predict + dream)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from worlds1k.data import SyntheticVideoDataset
from worlds1k.model.decoder import PixelDecoder
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


class _FlatEncoder(torch.nn.Module):
    def __init__(self, d_out: int, img_size: int = 16):
        super().__init__()
        self.proj = torch.nn.Linear(3 * img_size * img_size, d_out)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        b, t = video.shape[:2]
        return self.proj(video.reshape(b, t, -1))


class TestPixelDecoder:
    def test_forward_shape(self):
        dec = PixelDecoder(16, frame_height=16, frame_width=16)
        z = torch.randn(2, 16)
        out = dec(z)
        assert out.shape == (2, 3, 16, 16)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_decoder_training(self):
        from worlds1k.training.decode import DecoderTrainer, DecodeTrainConfig

        config = _small_config()
        model = WorldModel.from_config(config)
        encoder = _FlatEncoder(32)

        # Pretrain briefly so model produces meaningful latents
        ds = SyntheticVideoDataset(window_size=8, image_size=16)
        loader = DataLoader(ds, batch_size=2)

        decoder = PixelDecoder(config.d_latents[0], frame_height=16, frame_width=16)

        trainer = DecoderTrainer(
            model,
            encoder,
            decoder,
            loader,
            config=DecodeTrainConfig(max_frames=64, eval_freq=1),
        )
        result = trainer.train()
        assert len(result.train_losses) > 0


class TestDreamer:
    def test_dream(self):
        from worlds1k.inference.dream import Dreamer

        config = _small_config()
        model = WorldModel.from_config(config)
        model.eval()

        dreamer = Dreamer(model)
        result = dreamer.dream(torch.randn(1, 8, 32), num_steps=5)

        assert "z_trajectory" in result
        assert result["z_trajectory"].shape[1] == 6  # seed + 5 steps

    def test_dream_with_decoder(self):
        from worlds1k.inference.dream import Dreamer

        config = _small_config()
        model = WorldModel.from_config(config)
        decoder = PixelDecoder(config.d_latents[0], frame_height=16, frame_width=16)

        dreamer = Dreamer(model, decoder)
        result = dreamer.dream(torch.randn(1, 8, 32), num_steps=3)

        assert "frames_trajectory" in result
        assert result["frames_trajectory"].shape == (1, 4, 3, 16, 16)  # seed + 3 steps
