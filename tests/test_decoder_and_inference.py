"""Tests for diffusion decoder (phase 2) and inference (dream)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from worlds1k.data import SyntheticVideoDataset
from worlds1k.model.configs import build_diffusion_decoder
from worlds1k.model.diffusion_decoder import AdaGNDiffusionDecoder
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
        "image_size": 32,
        "image_channels": 3,
    }
    defaults.update(overrides)
    return WorldModelConfig(**defaults)


class _FlatEncoder(torch.nn.Module):
    def __init__(self, d_out: int, img_size: int = 32):
        super().__init__()
        self.proj = torch.nn.Linear(3 * img_size * img_size, d_out)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        b, t = video.shape[:2]
        return self.proj(video.reshape(b, t, -1))


class TestDiffusionDecoderForward:
    def test_forward_returns_loss(self):
        dec = AdaGNDiffusionDecoder(d_latent=16, image_size=32, d_model=32, num_train_timesteps=10)
        x = torch.rand(2, 3, 32, 32)
        z = torch.randn(2, 16)
        out = dec(x, z)
        assert "loss" in out
        assert out["loss"].shape == ()
        assert out["loss"].item() > 0

    def test_forward_backward(self):
        dec = AdaGNDiffusionDecoder(d_latent=16, image_size=32, d_model=32, num_train_timesteps=10)
        x = torch.rand(2, 3, 32, 32)
        z = torch.randn(2, 16)
        out = dec(x, z)
        out["loss"].backward()
        # Check gradients exist on decoder parameters
        grad_count = sum(1 for p in dec.parameters() if p.grad is not None)
        assert grad_count > 0


class TestDiffusionDecoderSample:
    def test_sample_shape(self):
        dec = AdaGNDiffusionDecoder(
            d_latent=16, image_size=32, d_model=32, num_train_timesteps=10, num_inference_steps=2
        )
        dec.eval()
        z = torch.randn(2, 16)
        frames = dec.sample(z)
        assert frames.shape == (2, 3, 32, 32)
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_sample_single_batch(self):
        dec = AdaGNDiffusionDecoder(
            d_latent=16, image_size=32, d_model=32, num_train_timesteps=10, num_inference_steps=2
        )
        dec.eval()
        z = torch.randn(1, 16)
        frames = dec.sample(z)
        assert frames.shape == (1, 3, 32, 32)


class TestBuildDiffusionDecoder:
    def test_factory_adagn(self):
        config = _small_config()
        dec = build_diffusion_decoder(config, arch="adagn", size="small", num_inference_steps=5)
        assert isinstance(dec, AdaGNDiffusionDecoder)
        assert dec.d_latent == 32  # d_input, not d_latent
        assert dec.image_size == 32

    def test_factory_default_is_adagn(self):
        config = _small_config()
        dec = build_diffusion_decoder(config, size="small", num_inference_steps=5)
        assert isinstance(dec, AdaGNDiffusionDecoder)

    def test_factory_forward(self):
        config = _small_config()
        dec = build_diffusion_decoder(config, arch="adagn", size="small", num_inference_steps=5, num_train_timesteps=10)
        x = torch.rand(1, 3, 32, 32)
        z = torch.randn(1, config.d_input)
        out = dec(x, z)
        assert "loss" in out


class TestDiffusionDecoderTrainer:
    def test_training_loop(self):
        from worlds1k.train.decoder import DiffusionDecoderTrainConfig, DiffusionDecoderTrainer

        config = _small_config()
        encoder = _FlatEncoder(config.d_input, img_size=32)

        ds = SyntheticVideoDataset(window_size=8, image_size=32)
        loader = DataLoader(ds, batch_size=2)

        decoder = AdaGNDiffusionDecoder(
            d_latent=config.d_input, image_size=32, d_model=32, num_train_timesteps=10
        )

        trainer = DiffusionDecoderTrainer(
            encoder,
            decoder,
            loader,
            config=DiffusionDecoderTrainConfig(max_frames=64, eval_freq=1),
        )
        result = trainer.train()
        assert len(result.train_losses) > 0


class TestDreamer:
    def test_dream_latents_only(self):
        from worlds1k.inference.dream import Dreamer

        config = _small_config()
        model = WorldModel.from_config(config)
        model.eval()

        dreamer = Dreamer(model)
        result = dreamer.dream(torch.randn(1, 8, 32), num_steps=5)

        assert "z_trajectory" in result
        assert result["z_trajectory"].shape[1] == 6  # seed + 5 steps

    def test_dream_with_diffusion_decoder(self):
        from worlds1k.inference.dream import Dreamer

        config = _small_config()
        model = WorldModel.from_config(config)
        decoder = AdaGNDiffusionDecoder(
            d_latent=config.d_input, image_size=32, d_model=32,
            num_train_timesteps=10, num_inference_steps=2,
        )

        dreamer = Dreamer(model, decoder)
        result = dreamer.dream(torch.randn(1, 8, 32), num_steps=3)

        assert "frames_trajectory" in result
        assert result["frames_trajectory"].shape == (1, 4, 3, 32, 32)  # seed + 3 steps
