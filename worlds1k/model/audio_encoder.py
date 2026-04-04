"""Audio encoder: converts raw audio into feature vectors for the world model.

Wraps a pretrained Whisper encoder backbone as a frozen audio feature extractor.
The backbone parameters are fixed; only a learned linear projection layer
is trained to map Whisper's output space into the dimensionality expected
by the world model.

Pipeline:

    Mel spectrogram (B, n_mels, T_audio)
        → Whisper encoder (frozen)  → encoder output (B, T_enc, d_backbone)
        → mean pool over time       → (B, d_backbone)
        → learned projection        → (B, d_output)
        → WorldModel level 1

AudioVideoEncoder combines visual and audio streams via early fusion:

    Video clip (B, T, C, H, W) + Audio features (B, T, n_mels, T_audio)
        → VisionEncoder per frame    → (B, T, d_visual)
        → AudioEncoder per window   → (B, T, d_audio)
        → concatenate               → (B, T, d_visual + d_audio)
        → WorldModel.forward()
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from transformers import WhisperModel

from worlds1k.model.encoder_base import (
    BaseAudioEncoder,
    BaseMultimodalEncoder,
    BaseVisionEncoder,
)

# Whisper model name → (HuggingFace repo, encoder hidden dimension)
_WHISPER_REGISTRY: dict[str, tuple[str, int]] = {
    "whisper-tiny": ("openai/whisper-tiny", 384),
    "whisper-base": ("openai/whisper-base", 512),
    "whisper-small": ("openai/whisper-small", 768),
}


class AudioEncoder(BaseAudioEncoder):
    """Frozen Whisper encoder backbone with a learnable output projection.

    Loads a pretrained Whisper encoder and freezes all of its parameters.
    A single trainable linear layer (preceded by LayerNorm) projects the
    mean-pooled encoder output into the dimensionality expected by the
    world model.

    Parameters
    ----------
    model_name : str
        Whisper variant to use.  One of ``"whisper-tiny"`` (384-d),
        ``"whisper-base"`` (512-d, default), or ``"whisper-small"`` (768-d).
    d_output : int
        Output feature dimensionality (default 256 — smaller than visual
        features since audio is lower bandwidth).
    backbone_frozen : bool
        Whether to freeze all backbone parameters (default ``True``).
        When ``True`` only the projection layer receives gradients.
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = list(_WHISPER_REGISTRY.keys())

    def __init__(
        self,
        model_name: str = "whisper-small",
        d_output: int = 256,
        *,
        backbone_frozen: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in _WHISPER_REGISTRY:
            raise ValueError(f"Unknown model_name {model_name!r}. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self._d_output = d_output
        self.backbone_frozen = backbone_frozen

        repo_id, d_backbone = _WHISPER_REGISTRY[model_name]
        self.d_backbone = d_backbone

        # Load pretrained Whisper model and keep only the encoder
        whisper = WhisperModel.from_pretrained(repo_id)
        self.backbone: nn.Module = whisper.encoder

        if backbone_frozen:
            self._freeze_backbone()

        # Trainable projection: d_backbone → d_output
        self.projection = nn.Sequential(
            nn.LayerNorm(d_backbone),
            nn.Linear(d_backbone, d_output),
        )

    @property
    def d_output(self) -> int:  # type: ignore[override]
        """Output feature dimensionality."""
        return self._d_output

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters so only the projection trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "whisper-small",
        d_output: int = 256,
        *,
        backbone_frozen: bool = True,
    ) -> AudioEncoder:
        """Create an AudioEncoder with a pretrained Whisper backbone.

        This is a convenience constructor that mirrors the ``from_pretrained``
        pattern used elsewhere in the codebase.  It is functionally identical
        to the normal ``__init__`` but makes the intent explicit when loading
        pretrained weights.

        Parameters
        ----------
        model_name : str
            Whisper variant (``"whisper-tiny"``, ``"whisper-base"``, or
            ``"whisper-small"``).
        d_output : int
            Output dimensionality (default 256).
        backbone_frozen : bool
            Whether to freeze backbone parameters (default ``True``).

        Returns
        -------
        AudioEncoder
            Initialised encoder with pretrained backbone loaded.
        """
        return cls(model_name, d_output, backbone_frozen=backbone_frozen)

    def train(self, mode: bool = True) -> AudioEncoder:
        """Override train to keep the frozen backbone in eval mode."""
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        """Encode a batch of audio segments into feature vectors.

        Parameters
        ----------
        mel_features : torch.Tensor
            Batch of log-Mel spectrograms, shape ``(B, n_mels, T_audio)``.
            Values should match the preprocessing expected by Whisper
            (80-bin log-Mel, 30-second windows at 16 kHz).

        Returns
        -------
        torch.Tensor
            Projected feature vectors, shape ``(B, d_output)``.
        """
        # Run through frozen backbone — no gradient computation needed
        if self.backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(mel_features)
        else:
            outputs = self.backbone(mel_features)

        # Mean-pool over the time dimension to get a single vector per sample
        hidden_states = outputs.last_hidden_state  # (B, T_enc, d_backbone)
        pooled = hidden_states.mean(dim=1)  # (B, d_backbone)

        # Project to world-model dimensionality (this IS trainable)
        return self.projection(pooled)  # (B, d_output)


class AudioVideoEncoder(BaseMultimodalEncoder):
    """Early-fusion multimodal encoder combining visual and audio streams.

    Encodes video frames through a :class:`BaseVisionEncoder` and audio
    segments through a :class:`BaseAudioEncoder`, then concatenates the
    resulting feature vectors along the channel dimension to produce a
    single fused representation per time step.

    Output shape: ``(B, T, d_visual + d_audio)`` — passed directly to
    :meth:`WorldModel.forward`.

    Parameters
    ----------
    vision_encoder : BaseVisionEncoder
        Per-frame visual encoder (shared across time steps).
    audio_encoder : BaseAudioEncoder
        Per-window audio encoder (shared across time steps).
    """

    def __init__(
        self,
        vision_encoder: BaseVisionEncoder,
        audio_encoder: BaseAudioEncoder,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder

    @classmethod
    def from_pretrained(
        cls,
        visual_model_name: str = "dinov2-small",
        d_visual: int = 512,
        audio_model_name: str = "whisper-small",
        d_audio: int = 256,
        *,
        backbone_frozen: bool = True,
    ) -> AudioVideoEncoder:
        """Create an AudioVideoEncoder with pretrained visual and audio backbones.

        Parameters
        ----------
        visual_model_name : str
            DINOv2 variant (``"dinov2-small"``, ``"dinov2-base"``, or
            ``"dinov2-large"``).
        d_visual : int
            Visual output dimensionality (default 512).
        audio_model_name : str
            Whisper variant (``"whisper-tiny"``, ``"whisper-base"``, or
            ``"whisper-small"``).
        d_audio : int
            Audio output dimensionality (default 256).
        backbone_frozen : bool
            Whether to freeze both backbone parameters (default ``True``).

        Returns
        -------
        AudioVideoEncoder
            Initialised multimodal encoder ready for use.
        """
        from worlds1k.model.vision_encoder import VisionEncoder

        vision_enc = VisionEncoder.from_pretrained(visual_model_name, d_visual, backbone_frozen=backbone_frozen)
        audio_enc = AudioEncoder.from_pretrained(audio_model_name, d_audio, backbone_frozen=backbone_frozen)
        return cls(vision_enc, audio_enc)

    @property
    def d_output(self) -> int:  # type: ignore[override]
        """Total output feature dimensionality (visual + audio, per frame)."""
        return self.vision_encoder.d_output + self.audio_encoder.d_output

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode aligned video and audio into fused per-frame features.

        Parameters
        ----------
        video : torch.Tensor
            Batch of video clips, shape ``(B, T, C, H, W)``.
        audio : torch.Tensor
            Batch of per-frame audio segments, shape
            ``(B, T, n_mels, T_audio)``.  Each ``audio[:, t]`` is the
            log-Mel spectrogram corresponding to video frame ``t``.

        Returns
        -------
        torch.Tensor
            Fused feature vectors, shape ``(B, T, d_visual + d_audio)``.
            This tensor can be passed directly to
            :meth:`WorldModel.forward`.
        """
        B, T = video.shape[:2]

        # --- Visual stream ---
        visual_features = self.vision_encoder.encode_video(video)  # (B, T, d_visual)

        # --- Audio stream ---
        _, _, n_mels, T_audio = audio.shape
        audio_flat = audio.reshape(B * T, n_mels, T_audio)  # (B*T, n_mels, T_audio)
        audio_features = self.audio_encoder(audio_flat)  # (B*T, d_audio)
        audio_features = audio_features.view(B, T, -1)  # (B, T, d_audio)

        # --- Early fusion: concatenate along feature dimension ---
        return torch.cat([visual_features, audio_features], dim=-1)  # (B, T, d_v+d_a)
