"""Video dataset registry and streaming loader.

Provides a registry of known HuggingFace video datasets and a true
streaming :class:`StreamingVideoDataset` (``IterableDataset``) that
decodes clips on the fly. After the first epoch, decoded tensors are
cached to disk so subsequent epochs are instant (no re-downloading).

Quick start::

    from worlds1k.data import StreamingVideoDataset, list_datasets

    print(list_datasets())
    ds = StreamingVideoDataset("disney", max_samples=100)

    # With audio (for datasets that have audio tracks):
    ds = StreamingVideoDataset("epic-kitchens", max_samples=5, with_audio=True)

Each sample is a tuple ``(video,)`` or ``(video, audio)`` where video is
a ``(T, C, H, W)`` float tensor in ``[0, 1]`` and audio is a
``(T, n_mels, T_audio)`` mel spectrogram tensor.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
WHISPER_N_MELS = 80
WHISPER_CHUNK_SECONDS = 30
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS  # 480000


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a HuggingFace video dataset."""

    hf_path: str
    video_column: str = "video"
    default_split: str = "train"
    description: str = ""
    gated: bool = False


REGISTRY: dict[str, DatasetSpec] = {
    "ucf101": DatasetSpec(
        "sayakpaul/ucf101-subset",
        description="UCF101 action recognition subset -- tiny, ideal for smoke tests.",
    ),
    "disney": DatasetSpec(
        "Wild-Heart/Disney-VideoGeneration-Dataset",
        description="Disney animated video clips (640x360).",
    ),
    "open-sora": DatasetSpec(
        "LanguageBind/Open-Sora-Plan-v1.1.0",
        video_column="mp4",
        description="Open-Sora Plan v1.1 video generation clips (1080p).",
    ),
    "kinetics400-sample": DatasetSpec(
        "JackWong0911/kinetic-400_450samples",
        video_column="mp4",
        description="Kinetics-400 sample -- 450 clips with raw mp4 bytes.",
    ),
    "epic-kitchens": DatasetSpec(
        "awsaf49/epic_kitchens_100",
        video_column="__hf_files__",
        description="EPIC-KITCHENS-100 -- 268 kitchen videos with audio (501 GB, file-based).",
    ),
    "finevideo": DatasetSpec(
        "HuggingFaceFV/finevideo",
        video_column="mp4",
        description="43K YouTube videos with rich annotations (~3 400 h).",
        gated=True,
    ),
    "egocentric-10k": DatasetSpec(
        "builddotai/Egocentric-10K",
        video_column="mp4",
        description="10K egocentric factory video samples (1080p).",
        gated=True,
    ),
}


def list_datasets() -> dict[str, DatasetSpec]:
    """Return the full dataset registry."""
    return dict(REGISTRY)


def resolve_dataset(name: str) -> DatasetSpec:
    """Look up *name* in the registry, or treat it as a raw HuggingFace path."""
    if name in REGISTRY:
        return REGISTRY[name]
    return DatasetSpec(hf_path=name)


def _process_video(source: Any, window_size: int, image_size: int) -> torch.Tensor:
    """Decode, window-sample, and resize a video.

    Returns a ``(T, C, H, W)`` float tensor in ``[0, 1]``.
    """
    from torchcodec.decoders import VideoDecoder

    if isinstance(source, bytes):
        decoder = VideoDecoder(source)
    elif isinstance(source, VideoDecoder) or (hasattr(source, "metadata") and hasattr(source.metadata, "num_frames")):
        decoder = source
    else:
        msg = f"Unsupported video source type: {type(source).__name__}"
        raise TypeError(msg)

    n_frames = decoder.metadata.num_frames

    if n_frames >= window_size:
        start = torch.randint(0, n_frames - window_size + 1, (1,)).item()
        video = decoder.get_frames_in_range(start=start, stop=start + window_size).data
    else:
        video = decoder.get_frames_in_range(start=0, stop=n_frames).data
        pad = window_size - n_frames
        video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)

    video = video.float() / 255.0

    if video.shape[-2] != image_size or video.shape[-1] != image_size:
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)

    return video


def _process_audio_video(
    raw_bytes: bytes, window_size: int, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode video + audio from raw bytes, return aligned (video, audio) pair.

    Audio is extracted as per-frame mel spectrograms compatible with Whisper:
    ``(T, 80, 3000)`` where each frame gets a 30s Whisper-sized context window
    centered on that frame's timestamp. Frames share overlapping audio context.

    For efficiency, we compute one mel spectrogram for the entire clip duration
    and slice per-frame windows from it.
    """
    from torchcodec.decoders import AudioDecoder, VideoDecoder
    from transformers import WhisperFeatureExtractor

    vdec = VideoDecoder(raw_bytes)
    adec = AudioDecoder(raw_bytes, sample_rate=WHISPER_SAMPLE_RATE)

    n_frames = vdec.metadata.num_frames
    fps = vdec.metadata.average_fps

    start_frame = torch.randint(0, n_frames - window_size + 1, (1,)).item() if n_frames >= window_size else 0

    end_frame = min(start_frame + window_size, n_frames)
    video = vdec.get_frames_in_range(start=start_frame, stop=end_frame).data
    if video.size(0) < window_size:
        pad = window_size - video.size(0)
        video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
    video = video.float() / 255.0
    if video.shape[-2] != image_size or video.shape[-1] != image_size:
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)

    # Get all audio samples as mono
    all_audio = adec.get_all_samples().data  # (channels, total_samples)
    all_audio = all_audio.mean(dim=0) if all_audio.size(0) > 1 else all_audio.squeeze(0)

    # Compute mel spectrogram for per-frame windows
    feature_extractor = WhisperFeatureExtractor()
    mel_list = []

    for i in range(window_size):
        frame_idx = start_frame + i
        # Center a 30s window on this frame's timestamp
        frame_time = frame_idx / fps
        center_sample = int(frame_time * WHISPER_SAMPLE_RATE)
        half_window = WHISPER_N_SAMPLES // 2
        audio_start = max(0, center_sample - half_window)
        audio_end = audio_start + WHISPER_N_SAMPLES

        if audio_end > all_audio.size(0):
            audio_end = all_audio.size(0)
            audio_start = max(0, audio_end - WHISPER_N_SAMPLES)

        chunk = all_audio[audio_start:audio_end]
        # Pad if too short
        if chunk.size(0) < WHISPER_N_SAMPLES:
            chunk = F.pad(chunk, (0, WHISPER_N_SAMPLES - chunk.size(0)))

        mel = feature_extractor(chunk.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        mel_list.append(mel.input_features.squeeze(0))  # (80, 3000)

    audio_tensor = torch.stack(mel_list)  # (T, 80, 3000)

    return video, audio_tensor


def _cache_key(
    name: str, split: str, max_samples: int | None, window_size: int, image_size: int, with_audio: bool
) -> str:
    raw = f"{name}|{split}|{max_samples}|{window_size}|{image_size}|{with_audio}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class StreamingVideoDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    """Streaming video dataset with automatic disk caching.

    First epoch: streams from HuggingFace, decodes clips, saves each as a
    ``.pt`` file in ``cache_dir``. Subsequent epochs: reads directly from
    cache.

    When ``with_audio=True``, also decodes the audio track and returns
    ``(video, audio)`` tuples where audio is per-frame mel spectrograms
    compatible with the Whisper encoder.

    Parameters
    ----------
    name_or_path : str
        Short registry name or full HuggingFace path.
    max_samples : int or None
        Stop after this many clips per epoch.
    window_size : int
        Frames per clip.
    image_size : int
        Spatial resolution (height = width).
    split : str or None
        Override the default split.
    token : str or None
        HuggingFace API token.
    with_audio : bool
        If True, decode audio track and return (video, audio) tuples.
    cache_dir : str or Path or None
        Cache directory. ``"auto"`` = ``~/.cache/worlds1k/<hash>``.
    """

    def __init__(
        self,
        name_or_path: str,
        *,
        max_samples: int | None = 100,
        window_size: int = 128,
        image_size: int = 64,
        split: str | None = None,
        token: str | None = None,
        with_audio: bool = False,
        cache_dir: str | Path | None = "auto",
    ) -> None:
        self._spec = resolve_dataset(name_or_path)
        self._split = split or self._spec.default_split
        self._max_samples = max_samples
        self._window_size = window_size
        self._image_size = image_size
        self._token = token
        self._with_audio = with_audio

        if cache_dir == "auto":
            key = _cache_key(name_or_path, self._split, max_samples, window_size, image_size, with_audio)
            self._cache_dir = Path.home() / ".cache" / "worlds1k" / key
        elif cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = None

    @property
    def _cached_clips(self) -> list[Path]:
        if self._cache_dir is None:
            return []
        return sorted(self._cache_dir.glob("clip_*.pt"))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        cached = self._cached_clips
        if cached:
            print(f"loading {len(cached)} cached clips from {self._cache_dir}")  # noqa: T201
            yield from self._iter_cache(cached)
        else:
            yield from self._iter_and_cache()

    def _iter_cache(self, files: list[Path]) -> Iterator[tuple[torch.Tensor, ...]]:
        indices = torch.randperm(len(files)).tolist()
        for i in indices:
            data = torch.load(files[i], weights_only=True, map_location="cpu")
            if isinstance(data, dict):
                yield (data["video"], data["audio"])
            else:
                yield (data,)

    def _iter_and_cache(self) -> Iterator[tuple[torch.Tensor, ...]]:
        spec = self._spec
        source = self._raw_iter_hf_files() if spec.video_column == "__hf_files__" else self._raw_iter_stream()

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        for item in source:
            if self._cache_dir is not None:
                torch.save(item, self._cache_dir / f"clip_{idx:06d}.pt")
            idx += 1
            if isinstance(item, dict):
                yield (item["video"], item["audio"])
            else:
                yield (item,)

        if self._cache_dir is not None and idx > 0:
            print(f"cached {idx} clips to {self._cache_dir}")  # noqa: T201

    def _raw_iter_stream(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        from datasets import load_dataset

        spec = self._spec
        stream = load_dataset(spec.hf_path, split=self._split, streaming=True, token=self._token)
        col = spec.video_column
        yielded = 0

        for item in stream:
            if self._max_samples is not None and yielded >= self._max_samples:
                break
            try:
                if self._with_audio and isinstance(item[col], bytes):
                    video, audio = _process_audio_video(item[col], self._window_size, self._image_size)
                    yield {"video": video, "audio": audio}
                else:
                    yield _process_video(item[col], self._window_size, self._image_size)
            except Exception:  # noqa: S112
                continue
            yielded += 1

    def _raw_iter_hf_files(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        from huggingface_hub import HfFileSystem

        spec = self._spec
        fs = HfFileSystem(token=self._token)
        paths = fs.glob(f"datasets/{spec.hf_path}/**/*.MP4")
        if not paths:
            paths = fs.glob(f"datasets/{spec.hf_path}/**/*.mp4")
        yielded = 0

        for path in paths:
            if self._max_samples is not None and yielded >= self._max_samples:
                break
            try:
                with fs.open(path, "rb") as f:
                    raw = f.read()
                if self._with_audio:
                    video, audio = _process_audio_video(raw, self._window_size, self._image_size)
                    yield {"video": video, "audio": audio}
                else:
                    yield _process_video(raw, self._window_size, self._image_size)
            except Exception:  # noqa: S112
                continue
            yielded += 1


class SyntheticVideoDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    """Random video tensors for verifying the training pipeline."""

    def __init__(
        self,
        num_samples: int = 100,
        window_size: int = 128,
        image_size: int = 64,
        channels: int = 3,
        *,
        include_audio: bool = False,
        n_mels: int = 80,
        audio_time_steps: int = 3000,
    ) -> None:
        self.num_samples = num_samples
        self.window_size = window_size
        self.image_size = image_size
        self.channels = channels
        self.include_audio = include_audio
        self.n_mels = n_mels
        self.audio_time_steps = audio_time_steps

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        for _ in range(self.num_samples):
            video = torch.rand(self.window_size, self.channels, self.image_size, self.image_size)
            if self.include_audio:
                audio = torch.randn(self.window_size, self.n_mels, self.audio_time_steps)
                yield (video, audio)
            else:
                yield (video,)
