"""Visual encoder: converts raw RGB frames into feature vectors for the world model.

Wraps a pretrained DINOv2 backbone as a frozen visual feature extractor.
The backbone parameters are fixed; only a learned linear projection layer
is trained to map DINOv2's output space into the dimensionality expected
by the first WorldLayer (d_input).

Pipeline:

    RGB image (B, C, H, W)
        → ImageNet normalisation (registered buffers)
        → DINOv2 backbone (frozen)  → CLS token (B, d_backbone)
        → learned projection        → (B, d_output)
        → WorldModel level 1

VideoEncoder extends this to process temporal sequences:

    Video clip (B, T, C, H, W)
        → VisionEncoder.encode_video  → (B, T, d_output)
        → WorldModel.forward()
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from transformers import AutoModel

from worlds1k.model.encoder_base import BaseVisionEncoder

# ImageNet channel-wise statistics used by DINOv2 preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 model name → (HuggingFace repo, CLS token dimension)
_DINOV2_REGISTRY: dict[str, tuple[str, int]] = {
    "dinov2-small": ("facebook/dinov2-small", 384),
    "dinov2-base": ("facebook/dinov2-base", 768),
    "dinov2-large": ("facebook/dinov2-large", 1024),
}


class VisionEncoder(BaseVisionEncoder):
    """Frozen DINOv2 backbone with a learnable output projection.

    Loads a pretrained DINOv2 vision transformer and freezes all of its
    parameters.  A single trainable linear layer projects the CLS token
    embedding into the dimensionality expected by the world model.

    ImageNet normalisation is applied inside :meth:`forward` so callers
    can pass raw ``[0, 1]``-range tensors without external preprocessing.

    Parameters
    ----------
    model_name : str
        DINOv2 variant to use.  One of ``"dinov2-small"`` (384-d, default),
        ``"dinov2-base"`` (768-d), or ``"dinov2-large"`` (1024-d).
    d_output : int
        Output feature dimensionality (default 512, matching
        ``WorldModelConfig.d_input``).
    backbone_frozen : bool
        Whether to freeze all backbone parameters (default ``True``).
        When ``True`` only the projection layer receives gradients.
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = list(_DINOV2_REGISTRY.keys())

    def __init__(
        self,
        model_name: str = "dinov2-small",
        d_output: int = 512,
        *,
        backbone_frozen: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in _DINOV2_REGISTRY:
            raise ValueError(f"Unknown model_name {model_name!r}. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self._d_output = d_output
        self.backbone_frozen = backbone_frozen

        repo_id, d_backbone = _DINOV2_REGISTRY[model_name]
        self.d_backbone = d_backbone

        # ImageNet normalisation as registered buffers (move with .to())
        self.register_buffer(
            "pixel_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

        # Load pretrained DINOv2 backbone
        self.backbone: nn.Module = AutoModel.from_pretrained(repo_id)

        if backbone_frozen:
            self._freeze_backbone()

        # Trainable projection: d_backbone → d_output
        self.projection = nn.Sequential(
            nn.LayerNorm(d_backbone),
            nn.Linear(d_backbone, d_output),
        )

    @property
    def d_output(self) -> int:  # type: ignore[override]
        """Output feature dimensionality (per frame)."""
        return self._d_output

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters so only the projection trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "dinov2-small",
        d_output: int = 512,
        *,
        backbone_frozen: bool = True,
    ) -> VisionEncoder:
        """Create a VisionEncoder with a pretrained DINOv2 backbone.

        This is a convenience constructor that mirrors the ``from_pretrained``
        pattern used elsewhere in the codebase.  It is functionally identical
        to the normal ``__init__`` but makes the intent explicit when loading
        pretrained weights.

        Parameters
        ----------
        model_name : str
            DINOv2 variant (``"dinov2-small"``, ``"dinov2-base"``, or
            ``"dinov2-large"``).
        d_output : int
            Output dimensionality (default 512).
        backbone_frozen : bool
            Whether to freeze backbone parameters (default ``True``).

        Returns
        -------
        VisionEncoder
            Initialised encoder with pretrained backbone loaded.
        """
        return cls(model_name, d_output, backbone_frozen=backbone_frozen)

    def train(self, mode: bool = True) -> VisionEncoder:
        """Override train to keep the frozen backbone in eval mode."""
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into feature vectors.

        Parameters
        ----------
        images : torch.Tensor
            Batch of RGB images, shape ``(B, C, H, W)``.  Pixel values
            should be in ``[0, 1]`` range — ImageNet normalisation is
            applied internally.

        Returns
        -------
        torch.Tensor
            Projected feature vectors, shape ``(B, d_output)``.
        """
        # Apply ImageNet normalisation
        images = (images - self.pixel_mean) / self.pixel_std

        # Run through frozen backbone — no gradient computation needed
        if self.backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)

        # Extract the CLS token representation
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, d_backbone)

        # Project to world-model dimensionality (this IS trainable)
        return self.projection(cls_token)  # (B, d_output)


class VideoEncoder(nn.Module):
    """Process video sequences frame-by-frame through a :class:`BaseVisionEncoder`.

    This is a thin convenience wrapper — it delegates entirely to
    :meth:`BaseVisionEncoder.encode_video`, which is provided by the ABC
    for free.  Prefer calling ``vision_encoder.encode_video(video)``
    directly in new code.

    Parameters
    ----------
    vision_encoder : BaseVisionEncoder
        The per-frame visual encoder (shared across all time steps).
    """

    def __init__(self, vision_encoder: BaseVisionEncoder) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "dinov2-small",
        d_output: int = 512,
        *,
        backbone_frozen: bool = True,
    ) -> VideoEncoder:
        """Create a VideoEncoder with a fresh pretrained VisionEncoder.

        Parameters
        ----------
        model_name : str
            DINOv2 variant (``"dinov2-small"``, ``"dinov2-base"``, or
            ``"dinov2-large"``).
        d_output : int
            Output dimensionality (default 512).
        backbone_frozen : bool
            Whether to freeze backbone parameters (default ``True``).

        Returns
        -------
        VideoEncoder
            Initialised video encoder ready for use.
        """
        vision_enc = VisionEncoder.from_pretrained(model_name, d_output, backbone_frozen=backbone_frozen)
        return cls(vision_enc)

    @property
    def d_output(self) -> int:
        """Output feature dimensionality (per frame)."""
        return self.vision_encoder.d_output

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a batch of video clips into per-frame feature sequences.

        Parameters
        ----------
        video : torch.Tensor
            Batch of video clips, shape ``(B, T, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Per-frame features, shape ``(B, T, d_output)``.  This tensor
            can be passed directly to :meth:`WorldModel.forward`.
        """
        return self.vision_encoder.encode_video(video)
