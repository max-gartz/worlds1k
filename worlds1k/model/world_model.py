"""Hierarchical predictive world model.

Stacks N WorldLayer instances into a hierarchy where each level operates at a
progressively coarser timescale.  Higher levels send top-down context that
constrains lower-level predictions.  The full model is trained end-to-end
with a single combined loss:

    L = Σ_ℓ γ_ℓ E_t[‖z - ẑ‖²] + λ_s Σ_ℓ E_t[Ω(z)] + λ_a Σ_ℓ E_t[Ω(a)]

All encoders, predictors, decoders, and action inference networks receive
gradients from this single objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .world_layer import WorldLayer


@dataclass
class WorldModelConfig:
    """Hyperparameters for the hierarchical world model.

    Parameters
    ----------
    num_levels : int
        Number of hierarchical levels (e.g. 3).
    d_input : int
        Dimensionality of raw sensory input (flattened frame features for
        level 1, or output of a frozen backbone).
    d_latents : list[int]
        Latent dimensionality at each level, length ``num_levels``.
        Level 1 is largest (most detail), higher levels are smaller.
    d_actions : list[int]
        Action code dimensionality at each level, length ``num_levels``.
        Derived as latent_dim // 4 by default.
    temporal_strides : list[int]
        Temporal stride T_ℓ at each level, length ``num_levels``.
    sparsity_target : float
        Target activation fraction ρ for SDR penalty (e.g. 0.05).
    lambda_sparsity : float
        Weight λ_s for the latent sparsity penalty.
    lambda_action_sparsity : float
        Weight λ_a for the action code sparsity penalty.
    level_weights : list[float]
        Per-level loss weights γ_ℓ, length ``num_levels``.
    num_transformer_heads : int
        Number of attention heads in each encoder's transformer.
    num_transformer_layers : int
        Number of transformer blocks in each encoder.
    dropout : float
        Dropout rate for all submodules.
    backbone_name : str
        DINOv2 variant used by the visual encoder.  One of
        ``"dinov2-small"`` (384-d), ``"dinov2-base"`` (768-d), or
        ``"dinov2-large"`` (1024-d).
    backbone_frozen : bool
        Whether the visual backbone's parameters are frozen during
        training (default ``True``).  When frozen only the projection
        layer and the world model itself receive gradients.
    audio_backbone_name : str | None
        Whisper variant used by the audio encoder.  One of
        ``"whisper-tiny"`` (384-d), ``"whisper-base"`` (512-d), or
        ``"whisper-small"`` (768-d).  Set to ``None`` (default) to
        disable the audio stream entirely.
    audio_backbone_frozen : bool
        Whether the audio backbone's parameters are frozen during
        training (default ``True``).
    d_audio : int
        Output dimensionality of the audio projection head (default 256).
        Only used when ``audio_backbone_name`` is not ``None``.
    image_size : int
        Output image size for the pixel decoder.
    image_channels : int
        Number of image channels (e.g. 3 for RGB).
    """

    num_levels: int = 3
    d_input: int = 512
    backbone_name: str = "dinov2-small"
    backbone_frozen: bool = True
    audio_backbone_name: str | None = None
    audio_backbone_frozen: bool = True
    d_audio: int = 256
    d_latents: list[int] = field(default_factory=lambda: [256, 128, 64])
    d_actions: list[int] = field(default_factory=lambda: [64, 32, 16])
    temporal_strides: list[int] = field(default_factory=lambda: [1, 8, 8])
    sparsity_target: float = 0.05
    lambda_sparsity: float = 0.01
    lambda_action_sparsity: float = 0.01
    level_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    num_transformer_heads: int = 8
    num_transformer_layers: int = 4
    dropout: float = 0.1
    image_size: int = 64
    image_channels: int = 3


class WorldModel(nn.Module):
    """Multi-level hierarchical predictive world model.

    Wires together N :class:`WorldLayer` instances with top-down context flow.
    A single forward pass encodes all levels bottom-up, then computes
    predictions with top-down context flowing from the highest level
    downward.

    The temporal striding works as follows: level 0 (lowest) processes every
    input frame.  Level 1 groups ``T_1`` level-0 latents into windows and
    produces one latent per window.  Level 2 groups ``T_2`` level-1 latents,
    etc.  The total number of input frames must be divisible by the product
    of all temporal strides.

    Parameters
    ----------
    config : WorldModelConfig
        Full model configuration.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        levels: list[WorldLayer] = []
        for i in range(config.num_levels):
            d_in = config.d_input if i == 0 else config.d_latents[i - 1]
            d_lower = config.d_latents[i - 1] if i > 0 else None
            levels.append(
                WorldLayer(
                    d_input=d_in,
                    d_latent=config.d_latents[i],
                    d_action=config.d_actions[i],
                    temporal_stride=config.temporal_strides[i],
                    d_lower=d_lower,
                    num_heads=config.num_transformer_heads,
                    num_blocks=config.num_transformer_layers,
                    dropout=config.dropout,
                )
            )
        self.levels = nn.ModuleList(levels)

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> WorldModel:
        """Create a world model from a configuration object."""
        return cls(config)

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full hierarchical forward pass.

        The forward pass proceeds in two phases:

        1. **Bottom-up encoding**: Each level encodes its input into latent
           windows and infers actions between consecutive latents.
        2. **Top-down prediction**: Starting from the highest level, top-down
           context flows downward.  Each level's predictor outputs a delta
           correction that is added to the top-down baseline.

        Parameters
        ----------
        frames : torch.Tensor
            Input frame features, shape ``(B, T, d_input)`` where T is the
            total number of time steps (must be divisible by the product of
            all temporal strides).

        Returns
        -------
        dict[str, torch.Tensor]
            ``"loss"`` — combined scalar loss.
            ``"level_losses"`` — per-level prediction losses, shape ``(num_levels,)``.
            ``"z"`` — list of encoded latents per level.
        """
        B, T, D = frames.shape
        cfg = self.config

        # --- Phase 1: Bottom-up encoding ---
        # Build windows for each level and encode
        level_outputs: list[dict[str, torch.Tensor]] = []
        current_seq = frames  # (B, T_current, d_current)

        for i, level in enumerate(self.levels):
            stride = cfg.temporal_strides[i]

            if stride == 1:
                # Level with stride 1: each frame is its own "window" of size 1
                # We need at least 2 windows for action inference
                N = current_seq.size(1)
                windows = current_seq.unsqueeze(2)  # (B, N, 1, d_in)
            else:
                # Reshape sequence into non-overlapping windows of size stride
                T_curr = current_seq.size(1)
                N = T_curr // stride
                windows = current_seq[:, : N * stride, :].view(B, N, stride, -1)  # (B, N, stride, d_in)

            # Forward through the level (no context yet — added in phase 2)
            out = level(windows, context=None)
            level_outputs.append(out)

            # The encoded latents become the input sequence for the next level
            current_seq = out["z"]  # (B, N, d_latent)

        # --- Phase 2: Top-down prediction with context ---
        # Re-run predictions with top-down context flowing from highest to lowest.
        # The encoding (z) is already done; we just need to recompute predictions
        # with the top-down signal.
        level_losses = []
        total_loss = torch.tensor(0.0, device=frames.device)

        for i in reversed(range(cfg.num_levels)):
            level = self.levels[i]
            out = level_outputs[i]
            z_enc = out["z"]  # (B, N, d_latent)
            N = z_enc.size(1)

            # Get top-down context from the level above (if it exists)
            context: torch.Tensor | None = None
            if i + 1 < cfg.num_levels:
                upper_level = self.levels[i + 1]
                if upper_level.top_down is not None:
                    upper_z = level_outputs[i + 1]["z"]
                    # Use the most recent upper-level latent as context
                    context = upper_level.top_down(upper_z[:, -1, :])  # (B, d_latent_i)

            # Recompute predictions with top-down context
            z_curr = z_enc[:, :-1, :].reshape(B * (N - 1), -1)
            actions = out["a"].reshape(B * (N - 1), -1)

            ctx_expanded: torch.Tensor | None = None
            if context is not None:
                ctx_expanded = context.unsqueeze(1).expand(-1, N - 1, -1)
                ctx_expanded = ctx_expanded.reshape(B * (N - 1), -1)

            z_pred = level.predictor(z_curr, actions, ctx_expanded)
            z_pred = z_pred.view(B, N - 1, -1)

            # Store updated predictions
            level_outputs[i]["z_pred"] = z_pred

            # Prediction loss: ‖z_{t+T} - ẑ_{t+T}‖²
            z_target = z_enc[:, 1:, :]  # (B, N-1, d_latent)
            pred_loss = (z_target - z_pred).pow(2).mean()

            # Sparsity penalties
            z_sparsity = self._sparsity_penalty(z_enc.reshape(-1, z_enc.size(-1)))
            a_sparsity = self._sparsity_penalty(out["a"].reshape(-1, out["a"].size(-1)))

            level_loss = (
                cfg.level_weights[i] * pred_loss
                + cfg.lambda_sparsity * z_sparsity
                + cfg.lambda_action_sparsity * a_sparsity
            )
            level_losses.append(pred_loss.detach())
            total_loss = total_loss + level_loss

        # Reverse so index 0 = level 0
        level_losses.reverse()

        return {
            "loss": total_loss,
            "level_losses": torch.stack(level_losses),
            "z": [out["z"] for out in level_outputs],
        }

    def _sparsity_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the SDR sparsity penalty Ω(z).

        Encourages a target activation fraction ρ::

            Ω(z) = (‖z‖₁ / d - ρ)²

        Parameters
        ----------
        z : torch.Tensor
            Latent activations, shape ``(B, d)``.

        Returns
        -------
        torch.Tensor
            Scalar sparsity penalty.
        """
        rho = self.config.sparsity_target
        d = z.size(-1)
        mean_activation = z.abs().sum(dim=-1).mean() / d
        return (mean_activation - rho) ** 2
