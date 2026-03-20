"""Tests for worlds1k.data — dataset registry and streaming loaders."""

from __future__ import annotations

import pytest
import torch

from worlds1k.data import (
    REGISTRY,
    DatasetSpec,
    StreamingVideoDataset,
    SyntheticVideoDataset,
    list_datasets,
    resolve_dataset,
)


class TestRegistry:
    def test_list_datasets_non_empty(self):
        datasets = list_datasets()
        assert len(datasets) > 0
        assert "ucf101" in datasets

    def test_resolve_known_name(self):
        spec = resolve_dataset("ucf101")
        assert spec.hf_path == "sayakpaul/ucf101-subset"
        assert spec.video_column == "video"

    def test_resolve_raw_path(self):
        spec = resolve_dataset("some-org/some-dataset")
        assert spec.hf_path == "some-org/some-dataset"
        assert spec.video_column == "video"

    def test_all_registry_entries_valid(self):
        for name, spec in REGISTRY.items():
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(spec, DatasetSpec)
            assert "/" in spec.hf_path, f"{name}: hf_path must be org/name"
            assert len(spec.video_column) > 0
            assert len(spec.description) > 0


class TestSyntheticVideoDataset:
    def test_video_only(self):
        ds = SyntheticVideoDataset(num_samples=5, window_size=16, image_size=32)
        samples = list(ds)
        assert len(samples) == 5
        sample = samples[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 1
        (video,) = sample
        assert video.shape == (16, 3, 32, 32)
        assert video.min() >= 0.0
        assert video.max() <= 1.0

    def test_audio_video(self):
        ds = SyntheticVideoDataset(
            num_samples=3,
            window_size=8,
            image_size=16,
            include_audio=True,
            n_mels=80,
            audio_time_steps=100,
        )
        samples = list(ds)
        assert len(samples) == 3
        video, audio = samples[0]
        assert video.shape == (8, 3, 16, 16)
        assert audio.shape == (8, 80, 100)

    def test_dataloader_compatible(self):
        from torch.utils.data import DataLoader

        ds = SyntheticVideoDataset(num_samples=4, window_size=8, image_size=16)
        loader = DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        assert len(batch) == 1
        assert batch[0].shape == (2, 8, 3, 16, 16)

    def test_re_iterable(self):
        ds = SyntheticVideoDataset(num_samples=3, window_size=4, image_size=8)
        epoch1 = list(ds)
        epoch2 = list(ds)
        assert len(epoch1) == 3
        assert len(epoch2) == 3


@pytest.mark.integration
class TestStreamingVideoDataset:
    """Integration tests that stream real data from HuggingFace.

    These are slow and require network access.  Run with::

        uv run pytest -m integration
    """

    def test_ucf101(self):
        ds = StreamingVideoDataset("ucf101", max_samples=2, window_size=128, image_size=64, shuffle_buffer=0)
        samples = list(ds)
        assert len(samples) >= 1
        (video,) = samples[0]
        assert video.shape == (128, 3, 64, 64)
        assert video.dtype == torch.float32
        assert video.min() >= 0.0 and video.max() <= 1.0

    def test_disney(self):
        ds = StreamingVideoDataset("disney", max_samples=2, window_size=128, image_size=64, shuffle_buffer=0)
        samples = list(ds)
        assert len(samples) >= 1
        (video,) = samples[0]
        assert video.shape == (128, 3, 64, 64)

    def test_open_sora(self):
        ds = StreamingVideoDataset("open-sora", max_samples=1, window_size=128, image_size=64, shuffle_buffer=0)
        samples = list(ds)
        assert len(samples) == 1
        (video,) = samples[0]
        assert video.shape == (128, 3, 64, 64)

    def test_raw_hf_path(self):
        ds = StreamingVideoDataset("sayakpaul/ucf101-subset", max_samples=1, window_size=128, image_size=64)
        samples = list(ds)
        assert len(samples) >= 1
