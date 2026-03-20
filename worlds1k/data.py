"""Video dataset registry and streaming loader.

Streams video from HuggingFace, extracts random temporal windows,
and caches decoded tensors to disk. Yields indefinitely — the training
loop controls duration via a frame budget.

Quick start::

    from worlds1k.data import StreamingVideoDataset

    ds = StreamingVideoDataset("disney")
    for (video,) in ds:  # yields forever
        train_on(video)  # (128, 3, 64, 64)

    # With audio:
    ds = StreamingVideoDataset("epic-kitchens", with_audio=True)
    for video, audio in ds:  # yields forever
        train_on(video, audio)
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
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * 30  # 30s window


@dataclass(frozen=True)
class DatasetSpec:
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
    return dict(REGISTRY)


def resolve_dataset(name: str) -> DatasetSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    return DatasetSpec(hf_path=name)


def _process_video(source: Any, window_size: int, image_size: int) -> torch.Tensor:
    """Decode, random-window, resize. Returns (T, C, H, W) in [0, 1]."""
    from torchcodec.decoders import VideoDecoder

    if isinstance(source, bytes):
        decoder = VideoDecoder(source)
    elif isinstance(source, VideoDecoder) or (hasattr(source, "metadata") and hasattr(source.metadata, "num_frames")):
        decoder = source
    else:
        msg = f"Unsupported video source type: {type(source).__name__}"
        raise TypeError(msg)

    n = decoder.metadata.num_frames
    start = torch.randint(0, max(1, n - window_size + 1), (1,)).item() if n >= window_size else 0
    end = min(start + window_size, n)
    video = decoder.get_frames_in_range(start=start, stop=end).data

    if video.size(0) < window_size:
        video = torch.cat([video, video[-1:].expand(window_size - video.size(0), -1, -1, -1)], dim=0)

    video = video.float() / 255.0
    if video.shape[-2] != image_size or video.shape[-1] != image_size:
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return video


def _process_audio_video(raw_bytes: bytes, window_size: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode video + audio, return aligned (video, audio) pair."""
    from torchcodec.decoders import AudioDecoder, VideoDecoder
    from transformers import WhisperFeatureExtractor

    vdec = VideoDecoder(raw_bytes)
    adec = AudioDecoder(raw_bytes, sample_rate=WHISPER_SAMPLE_RATE)

    n = vdec.metadata.num_frames
    fps = vdec.metadata.average_fps
    start = torch.randint(0, max(1, n - window_size + 1), (1,)).item() if n >= window_size else 0
    end = min(start + window_size, n)

    video = vdec.get_frames_in_range(start=start, stop=end).data
    if video.size(0) < window_size:
        video = torch.cat([video, video[-1:].expand(window_size - video.size(0), -1, -1, -1)], dim=0)
    video = video.float() / 255.0
    if video.shape[-2] != image_size or video.shape[-1] != image_size:
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)

    all_audio = adec.get_all_samples().data
    all_audio = all_audio.mean(dim=0) if all_audio.size(0) > 1 else all_audio.squeeze(0)

    fe = WhisperFeatureExtractor()
    mels = []
    half = WHISPER_N_SAMPLES // 2
    for i in range(window_size):
        center = int((start + i) / fps * WHISPER_SAMPLE_RATE)
        a_start = max(0, center - half)
        a_end = a_start + WHISPER_N_SAMPLES
        if a_end > all_audio.size(0):
            a_end = all_audio.size(0)
            a_start = max(0, a_end - WHISPER_N_SAMPLES)
        chunk = all_audio[a_start:a_end]
        if chunk.size(0) < WHISPER_N_SAMPLES:
            chunk = F.pad(chunk, (0, WHISPER_N_SAMPLES - chunk.size(0)))
        mel = fe(chunk.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        mels.append(mel.input_features.squeeze(0))

    return video, torch.stack(mels)


def _cache_key(name: str, window_size: int, image_size: int, with_audio: bool) -> str:
    raw = f"{name}|{window_size}|{image_size}|{with_audio}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class StreamingVideoDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    """Infinite streaming video dataset with disk caching.

    Yields random windows from cached video clips forever. The training
    loop controls duration via a frame budget (``--max-frames``).

    First iteration through the source data caches decoded clips to disk.
    All subsequent iterations read from cache with random window re-sampling
    each time (different temporal crops from the same videos).

    Parameters
    ----------
    name_or_path : str
        Short registry name or full HuggingFace path.
    window_size : int
        Frames per clip.
    image_size : int
        Spatial resolution (height = width).
    split : str or None
        Override the default split.
    token : str or None
        HuggingFace API token.
    with_audio : bool
        If True, decode audio and return (video, audio) tuples.
    max_videos : int or None
        Max number of source videos to download/cache.
    cache_dir : str or Path or None
        Cache directory. ``"auto"`` = ``~/.cache/worlds1k/<hash>``.
    """

    def __init__(
        self,
        name_or_path: str,
        *,
        window_size: int = 128,
        image_size: int = 64,
        split: str | None = None,
        token: str | None = None,
        with_audio: bool = False,
        max_videos: int | None = None,
        cache_dir: str | Path | None = "auto",
    ) -> None:
        self._spec = resolve_dataset(name_or_path)
        self._split = split or self._spec.default_split
        self._window_size = window_size
        self._image_size = image_size
        self._token = token
        self._with_audio = with_audio
        self._max_videos = max_videos

        if cache_dir == "auto":
            key = _cache_key(name_or_path, window_size, image_size, with_audio)
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
            # Cache exists — yield from it forever in shuffled order
            while True:
                for i in torch.randperm(len(cached)).tolist():
                    data = torch.load(cached[i], weights_only=True, map_location="cpu")
                    if isinstance(data, dict):
                        yield (data["video"], data["audio"])
                    else:
                        yield (data,)
        else:
            # First pass: stream, cache each clip, yield immediately.
            # After first pass completes, loop from cache.
            yield from self._stream_and_cache()
            cached = self._cached_clips
            if not cached:
                msg = f"No clips cached from {self._spec.hf_path}"
                raise RuntimeError(msg)
            while True:
                for i in torch.randperm(len(cached)).tolist():
                    data = torch.load(cached[i], weights_only=True, map_location="cpu")
                    if isinstance(data, dict):
                        yield (data["video"], data["audio"])
                    else:
                        yield (data,)

    def _stream_and_cache(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Stream from source, cache each clip to disk, yield immediately."""
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

        if idx > 0:
            print(f"cached {idx} clips to {self._cache_dir}")  # noqa: T201

    def _raw_iter_stream(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        from datasets import load_dataset

        spec = self._spec
        stream = load_dataset(spec.hf_path, split=self._split, streaming=True, token=self._token)
        col = spec.video_column
        yielded = 0

        for item in stream:
            if self._max_videos is not None and yielded >= self._max_videos:
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
            if self._max_videos is not None and yielded >= self._max_videos:
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
    """Random video tensors — yields indefinitely."""

    def __init__(
        self,
        window_size: int = 128,
        image_size: int = 64,
        channels: int = 3,
        *,
        include_audio: bool = False,
        n_mels: int = 80,
        audio_time_steps: int = 3000,
    ) -> None:
        self.window_size = window_size
        self.image_size = image_size
        self.channels = channels
        self.include_audio = include_audio
        self.n_mels = n_mels
        self.audio_time_steps = audio_time_steps

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        while True:
            video = torch.rand(self.window_size, self.channels, self.image_size, self.image_size)
            if self.include_audio:
                audio = torch.randn(self.window_size, self.n_mels, self.audio_time_steps)
                yield (video, audio)
            else:
                yield (video,)
