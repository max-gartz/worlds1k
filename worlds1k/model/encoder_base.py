"""Abstract base classes for sensory encoders.

Defines the contracts that all frame, audio, and multimodal encoders must
satisfy so that the training script and :class:`WorldModel` remain agnostic
to the concrete backbone (DINOv2, Whisper, future alternatives, …).

Concrete implementations live in :mod:`vision_encoder` and
:mod:`audio_encoder`; this module provides only the ABCs plus lightweight
factory helpers that map a :class:`WorldModelConfig` to the right encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from worlds1k.model.world_model import WorldModelConfig


class BaseVisionEncoder(ABC, nn.Module):
    """Contract for single-frame visual encoders.

    Subclasses must implement :meth:`forward` (single-image encoding) and
    expose the output dimensionality via the :attr:`d_output` property.

    The concrete :meth:`encode_video` helper is provided for free — it
    reshapes a ``(B, T, C, H, W)`` video tensor into a batch of frames,
    calls :meth:`forward`, then restores the temporal dimension.
    """

    @property
    @abstractmethod
    def d_output(self) -> int:
        """Output feature dimensionality (per frame)."""
        ...

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            ``(B, C, H, W)`` RGB images (pre-processing is the
            responsibility of the concrete subclass).

        Returns
        -------
        torch.Tensor
            ``(B, d_output)`` feature vectors.
        """
        ...

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a batch of video clips frame-by-frame.

        Parameters
        ----------
        video : torch.Tensor
            ``(B, T, C, H, W)`` video clips.

        Returns
        -------
        torch.Tensor
            ``(B, T, d_output)`` per-frame features.
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        features = self.forward(frames)  # (B*T, d_output)
        return features.view(B, T, -1)


class BaseAudioEncoder(ABC, nn.Module):
    """Contract for audio-segment encoders.

    Subclasses must encode a single Mel spectrogram into a fixed-length
    feature vector and expose the output dimensionality.
    """

    @property
    @abstractmethod
    def d_output(self) -> int:
        """Output feature dimensionality."""
        ...

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a batch of audio segments.

        Parameters
        ----------
        audio : torch.Tensor
            ``(B, n_mels, T_audio)`` log-Mel spectrograms.

        Returns
        -------
        torch.Tensor
            ``(B, d_output)`` feature vectors.
        """
        ...


class BaseMultimodalEncoder(ABC, nn.Module):
    """Contract for joint video + audio encoders.

    Subclasses fuse visual and audio streams and return a single
    per-time-step feature tensor consumed by :class:`WorldModel`.
    """

    @property
    @abstractmethod
    def d_output(self) -> int:
        """Total output feature dimensionality (per time step)."""
        ...

    @abstractmethod
    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode aligned video and audio into fused features.

        Parameters
        ----------
        video : torch.Tensor
            ``(B, T, C, H, W)`` video clips.
        audio : torch.Tensor
            ``(B, T, n_mels, T_audio)`` per-frame audio segments.

        Returns
        -------
        torch.Tensor
            ``(B, T, d_output)`` fused feature vectors.
        """
        ...


def build_vision_encoder(config: WorldModelConfig) -> BaseVisionEncoder:
    """Instantiate the vision encoder specified by *config*.

    Parameters
    ----------
    config : WorldModelConfig
        Must contain ``backbone_name``, ``d_input``, and
        ``backbone_frozen``.

    Returns
    -------
    BaseVisionEncoder
        A concrete vision encoder ready for use.
    """
    from worlds1k.model.vision_encoder import VisionEncoder

    return VisionEncoder(
        config.backbone_name,
        config.d_input,
        backbone_frozen=config.backbone_frozen,
    )


def build_audio_encoder(config: WorldModelConfig) -> BaseAudioEncoder:
    """Instantiate the audio encoder specified by *config*.

    Parameters
    ----------
    config : WorldModelConfig
        Must contain ``audio_backbone_name``, ``d_audio``, and
        ``audio_backbone_frozen``.

    Returns
    -------
    BaseAudioEncoder
        A concrete audio encoder ready for use.

    Raises
    ------
    ValueError
        If ``config.audio_backbone_name`` is ``None``.
    """
    from worlds1k.model.audio_encoder import AudioEncoder

    if config.audio_backbone_name is None:
        raise ValueError("Cannot build an audio encoder when config.audio_backbone_name is None.")
    return AudioEncoder(
        config.audio_backbone_name,
        config.d_audio,
        backbone_frozen=config.audio_backbone_frozen,
    )


def build_multimodal_encoder(config: WorldModelConfig) -> BaseMultimodalEncoder:
    """Instantiate a multimodal (video + audio) encoder from *config*.

    Parameters
    ----------
    config : WorldModelConfig
        Must contain both visual and audio backbone settings.

    Returns
    -------
    BaseMultimodalEncoder
        A concrete multimodal encoder ready for use.
    """
    from worlds1k.model.audio_encoder import AudioVideoEncoder

    vision_enc = build_vision_encoder(config)
    audio_enc = build_audio_encoder(config)
    return AudioVideoEncoder(vision_enc, audio_enc)
