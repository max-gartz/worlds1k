"""Hierarchical predictive world model.

Stacks N WorldLayer instances into a hierarchy where each level operates at a
progressively coarser timescale.  Higher levels send top-down context that
constrains lower-level predictions.  The full model is trained end-to-end
with a single combined loss:

    L = Σ_ℓ γ_ℓ E_t[‖z - ẑ‖²] + λ_s Σ_ℓ E_t[Ω(z)] + λ_a Σ_ℓ E_t[Ω(a)]

All encoders, predictors, decoders, and action inference networks receive
gradients from this single objective.

Core math (for level ℓ with temporal stride T_ℓ):

    Encode:   z_t^(ℓ) = E^(ℓ)(z_{t-T+1}^(ℓ-1), ..., z_t^(ℓ-1))
    Action:   a_t^(ℓ) = f_a^(ℓ)(z_t^(ℓ), z_{t+T}^(ℓ))
    Predict:  ẑ_{t+T}^(ℓ) = P^(ℓ)(z_t^(ℓ), a_t^(ℓ)) + α · c^(ℓ+1)
    Loss:     ‖z_{t+T}^(ℓ) - ẑ_{t+T}^(ℓ)‖² + λ_s Ω(z) + λ_a Ω(a)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ===================================================================
# Layer components
# ===================================================================


class Encoder(nn.Module):
    """Temporal encoder: compresses T_ℓ lower-level states into one latent.

    Uses a small **causal transformer** over the temporal window of
    lower-level representations.  The transformer attends only to
    current and past positions within the window (causal mask), then
    a learned readout projects the final position's hidden state to
    the output latent z^(ℓ).  The fixed output size forces information
    compression across the temporal window.

    Architecture:
    1. Linear projection from d_input → d_model (transformer hidden dim).
    2. Learned positional embeddings for T_ℓ positions.
    3. Stack of causal self-attention + feedforward transformer blocks.
    4. Layer norm on the last position's output.
    5. Linear readout from d_model → d_latent.

    Parameters
    ----------
    d_input : int
        Dimensionality of each incoming lower-level state.
    d_latent : int
        Dimensionality of the output latent z^(ℓ).
    temporal_stride : int
        Number of lower-level steps consumed per one output (T_ℓ).
    d_model : int
        Hidden dimensionality of the causal transformer (default 128).
    num_heads : int
        Number of attention heads (default 4).
    num_blocks : int
        Number of transformer blocks (default 2).
    dropout : float
        Dropout rate (default 0.0).
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        *,
        temporal_stride: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.temporal_stride = temporal_stride
        self.d_model = d_model

        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Embedding(temporal_stride, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, d_latent)

        causal_mask = torch.triu(
            torch.ones(temporal_stride, temporal_stride, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, z_lower: torch.Tensor) -> torch.Tensor:
        """Encode a window of lower-level states into a single latent.

        Parameters
        ----------
        z_lower : torch.Tensor
            Lower-level states, shape ``(B, T_ℓ, d_input)``.

        Returns
        -------
        torch.Tensor
            Latent state z^(ℓ), shape ``(B, d_latent)``.
        """
        B, T, _ = z_lower.shape
        positions = torch.arange(T, device=z_lower.device)
        x = self.input_proj(z_lower) + self.pos_emb(positions)
        x = self.transformer(x, mask=self.causal_mask[:T, :T])
        x = self.norm(x[:, -1, :])
        return self.readout(x)


class ActionHead(nn.Module):
    """Infer the action (transition code) between consecutive latent states.

    The action a_t^(ℓ) captures *what changed* — it is not a motor command
    but a learned bottleneck that must be informative enough for prediction
    yet compact enough to generalize.  At level 1 this may learn optical
    flow; at higher levels it may learn abstract action concepts.

    Parameters
    ----------
    d_latent : int
        Dimensionality of each latent state z^(ℓ).
    d_action : int
        Dimensionality of the inferred action code.
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, d_latent: int, d_action: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_action = d_action
        d_hidden = d_latent
        self.net = nn.Sequential(
            nn.Linear(2 * d_latent, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_action),
        )

    def forward(self, z_curr: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """Infer the action code from two consecutive latent states.

        Parameters
        ----------
        z_curr : torch.Tensor
            Current latent, shape ``(B, d_latent)``.
        z_next : torch.Tensor
            Next latent (teacher-forced target), shape ``(B, d_latent)``.

        Returns
        -------
        torch.Tensor
            Action code a^(ℓ), shape ``(B, d_action)``.
        """
        return self.net(torch.cat([z_curr, z_next], dim=-1))


class Predictor(nn.Module):
    """Predict the next input from current latent state and action.

    Outputs in the level's **input space** (d_input), not latent space.
    This means predictions can be fed back through the encoder for
    autoregressive rollout.

    When top-down context is available::

        x̂_{t+T}^(ℓ) = α · c^(ℓ+1) + P^(ℓ)(z_t^(ℓ), a_t^(ℓ))

    Parameters
    ----------
    d_latent : int
        Dimensionality of the latent state z^(ℓ).
    d_action : int
        Dimensionality of the action code a^(ℓ).
    d_output : int
        Dimensionality of the prediction output (the level's input space).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, d_latent: int, d_action: int, d_output: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_action = d_action
        self.d_output = d_output
        self.alpha = nn.Parameter(torch.tensor(1.0))

        d_hidden = d_latent * 2
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
        )

    def forward(
        self,
        z_curr: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next input.

        Parameters
        ----------
        z_curr : torch.Tensor
            Current latent, shape ``(B, d_latent)``.
        action : torch.Tensor
            Inferred action code, shape ``(B, d_action)``.
        context : torch.Tensor or None
            Top-down context from level above, shape ``(B, d_output)``.
            ``None`` for the highest level.

        Returns
        -------
        torch.Tensor
            Predicted next input, shape ``(B, d_output)``.
        """
        delta = self.net(torch.cat([z_curr, action], dim=-1))
        if context is not None:
            return self.alpha * context + delta
        return delta


class TopDownDecoder(nn.Module):
    """Project a higher-level latent down as context for the level below.

    Maps z^(ℓ) into the dimensionality of level ℓ-1's **input space**
    so it can serve as the baseline prediction in the lower-level predictor.

    Parameters
    ----------
    d_upper : int
        Dimensionality of this level's latent z^(ℓ).
    d_lower_input : int
        Dimensionality of the level below's input space.
    """

    def __init__(self, d_upper: int, d_lower_input: int) -> None:
        super().__init__()
        self.d_upper = d_upper
        self.d_lower = d_lower_input
        self.proj = nn.Sequential(
            nn.Linear(d_upper, d_lower_input),
            nn.LayerNorm(d_lower_input),
        )

    def forward(self, z_upper: torch.Tensor) -> torch.Tensor:
        """Decode top-down context.

        Parameters
        ----------
        z_upper : torch.Tensor
            Higher-level latent, shape ``(B, d_upper)``.

        Returns
        -------
        torch.Tensor
            Context signal c^(ℓ) for the level below, shape ``(B, d_lower)``.
        """
        return self.proj(z_upper)


# ===================================================================
# WorldLayer — one complete hierarchical level
# ===================================================================


class WorldLayer(nn.Module):
    """One complete hierarchical world layer: encoder, action head, predictor, decoder.

    Bundles the four sub-modules that define a single layer of the
    predictive hierarchy and provides a unified forward pass that
    returns all intermediate representations needed for the combined loss.

    Parameters
    ----------
    d_input : int
        Dimensionality of each incoming lower-level state.
    d_latent : int
        Dimensionality of this level's latent representation.
    d_action : int
        Dimensionality of the inferred action code.
    temporal_stride : int
        Number of lower-level steps per one level step (T_ℓ).
    d_lower : int or None
        Dimensionality of the level below (for the top-down decoder).
        ``None`` for the lowest level, which has no top-down output.
    num_heads : int
        Number of attention heads in the encoder transformer (default 4).
    num_blocks : int
        Number of transformer blocks in the encoder (default 2).
    dropout : float
        Dropout rate for all submodules (default 0.1).
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        d_action: int,
        *,
        temporal_stride: int,
        d_lower: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            d_input,
            d_latent,
            temporal_stride=temporal_stride,
            d_model=d_latent,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.d_input = d_input
        self.latent_norm = nn.LayerNorm(d_latent)
        self.action_head = ActionHead(d_latent, d_action, dropout=dropout)
        self.predictor = Predictor(d_latent, d_action, d_output=d_input, dropout=dropout)
        self.top_down: TopDownDecoder | None = None
        if d_lower is not None:
            # d_lower is the input dim of the level below
            self.top_down = TopDownDecoder(d_latent, d_lower)

    def encode(self, z_lower_windows: torch.Tensor) -> torch.Tensor:
        """Encode temporal windows into latent vectors.

        Parameters
        ----------
        z_lower_windows : torch.Tensor
            Shape ``(B, N_windows, T_ℓ, d_input)``.

        Returns
        -------
        torch.Tensor
            Encoded latents ``(B, N_windows, d_latent)``.
        """
        B, N, T, D = z_lower_windows.shape
        z_flat = z_lower_windows.reshape(B * N, T, D)
        z_enc = self.encoder(z_flat)
        z_enc = self.latent_norm(z_enc)
        return z_enc.view(B, N, -1)

    def predict(self, z_prev: torch.Tensor, z_curr: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the next input from two consecutive latents.

        Parameters
        ----------
        z_prev : torch.Tensor
            Previous latent, shape ``(B, d_latent)``.
        z_curr : torch.Tensor
            Current latent, shape ``(B, d_latent)``.
        context : torch.Tensor or None
            Top-down context from level above, shape ``(B, d_input)``.

        Returns
        -------
        torch.Tensor
            Predicted next input, shape ``(B, d_input)``.
        """
        action = self.action_head(z_prev, z_curr)
        return self.predictor(z_curr, action, context)

    def forward(
        self,
        z_lower_windows: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a full level forward pass (encode + predict).

        Parameters
        ----------
        z_lower_windows : torch.Tensor
            Consecutive temporal windows of lower-level states,
            shape ``(B, N_windows, T_ℓ, d_input)`` where ``N_windows >= 2``.
        context : torch.Tensor or None
            Top-down context from the level above, shape ``(B, d_input)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"z"`` — encoded latents, shape ``(B, N_windows, d_latent)``
            ``"a"`` — action codes, shape ``(B, N_windows - 1, d_action)``
            ``"predicted"`` — predicted next inputs, shape ``(B, N_windows - 1, d_input)``
            ``"c_down"`` — top-down context for level below (if applicable)
        """
        B, N = z_lower_windows.shape[:2]

        z_enc = self.encode(z_lower_windows)

        # Infer actions between consecutive latents
        z_curr = z_enc[:, :-1, :].reshape(B * (N - 1), -1)
        z_next = z_enc[:, 1:, :].reshape(B * (N - 1), -1)
        actions = self.action_head(z_curr, z_next)
        actions = actions.view(B, N - 1, -1)

        # Predict next inputs (in this level's input space)
        ctx_expanded: torch.Tensor | None = None
        if context is not None:
            ctx_expanded = context.unsqueeze(1).expand(-1, N - 1, -1)
            ctx_expanded = ctx_expanded.reshape(B * (N - 1), -1)

        predicted = self.predictor(z_curr, actions.view(B * (N - 1), -1), ctx_expanded)
        predicted = predicted.view(B, N - 1, -1)

        result: dict[str, torch.Tensor] = {
            "z": z_enc,
            "a": actions,
            "predicted": predicted,
        }

        if self.top_down is not None:
            c_down = self.top_down(z_enc[:, -1, :])
            result["c_down"] = c_down

        return result


# ===================================================================
# WorldModelConfig
# ===================================================================


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
        DINOv2 variant used by the visual encoder.
    backbone_frozen : bool
        Whether the visual backbone's parameters are frozen during training.
    audio_backbone_name : str | None
        Whisper variant used by the audio encoder.  ``None`` to disable.
    audio_backbone_frozen : bool
        Whether the audio backbone's parameters are frozen during training.
    d_audio : int
        Output dimensionality of the audio projection head.
    image_size : int
        Output image size for the vision decoder.
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


# ===================================================================
# WorldModel — multi-level hierarchy
# ===================================================================


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
            # d_lower = input dim of the level below (for top-down context)
            d_lower = (config.d_input if i == 1 else config.d_latents[i - 2]) if i > 0 else None
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

    def encode(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Encode features into latents at all hierarchy levels.

        Parameters
        ----------
        features : torch.Tensor
            Input features, shape ``(B, T, d_input)``.

        Returns
        -------
        list[torch.Tensor]
            Encoded latents per level.
        """
        B = features.size(0)
        cfg = self.config
        current_seq = features
        all_z = []

        for i, level in enumerate(self.levels):
            stride = cfg.temporal_strides[i]
            if stride == 1:
                windows = current_seq.unsqueeze(2)
            else:
                T_curr = current_seq.size(1)
                N = T_curr // stride
                windows = current_seq[:, : N * stride, :].view(B, N, stride, -1)

            z = level.encode(windows)
            all_z.append(z)
            current_seq = z

        return all_z


    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
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
        B, T, D = features.shape
        cfg = self.config

        # --- Phase 1: Bottom-up encoding ---
        level_outputs: list[dict[str, torch.Tensor]] = []
        current_seq = features

        for i, level in enumerate(self.levels):
            stride = cfg.temporal_strides[i]

            if stride == 1:
                N = current_seq.size(1)
                windows = current_seq.unsqueeze(2)
            else:
                T_curr = current_seq.size(1)
                N = T_curr // stride
                windows = current_seq[:, : N * stride, :].view(B, N, stride, -1)

            out = level(windows, context=None)
            level_outputs.append(out)
            current_seq = out["z"]

        # --- Phase 2: Top-down prediction with context ---
        # Collect inputs for each level (for prediction loss in input space)
        level_inputs: list[torch.Tensor] = []
        current_seq = features
        for i in range(cfg.num_levels):
            level_inputs.append(current_seq)
            current_seq = level_outputs[i]["z"]

        level_losses = []
        z_sparsity_achieved = []
        a_sparsity_achieved = []
        alpha_values = []
        action_entropies = []
        total_loss = torch.tensor(0.0, device=features.device)

        for i in reversed(range(cfg.num_levels)):
            level = self.levels[i]
            out = level_outputs[i]
            z_enc = out["z"]
            N = z_enc.size(1)

            context: torch.Tensor | None = None
            if i + 1 < cfg.num_levels:
                upper_level = self.levels[i + 1]
                if upper_level.top_down is not None:
                    upper_z = level_outputs[i + 1]["z"]
                    context = upper_level.top_down(upper_z[:, -1, :])

            z_curr = z_enc[:, :-1, :].reshape(B * (N - 1), -1)
            actions = out["a"].reshape(B * (N - 1), -1)

            ctx_expanded: torch.Tensor | None = None
            if context is not None:
                ctx_expanded = context.unsqueeze(1).expand(-1, N - 1, -1)
                ctx_expanded = ctx_expanded.reshape(B * (N - 1), -1)

            # Predict next input (in this level's input space)
            predicted = level.predictor(z_curr, actions, ctx_expanded)
            predicted = predicted.view(B, N - 1, -1)

            level_outputs[i]["predicted"] = predicted

            # Loss: compare predicted input against actual next input
            # The inputs to this level are one-per-window, so the target
            # for prediction at position t is the input at position t+1
            inputs_for_level = level_inputs[i]
            stride = cfg.temporal_strides[i]
            if stride == 1:
                input_target = inputs_for_level[:, 1:N, :]
            else:
                # Each window produces one latent, target is the center/last of next window
                input_target = inputs_for_level[:, stride:N * stride:stride, :][:, :N - 1, :]

            pred_loss = (input_target - predicted).pow(2).mean()

            z_sparsity, z_act_frac = self._sparsity_penalty(z_enc.reshape(-1, z_enc.size(-1)))
            a_sparsity, a_act_frac = self._sparsity_penalty(out["a"].reshape(-1, out["a"].size(-1)))

            level_loss = (
                cfg.level_weights[i] * pred_loss
                + cfg.lambda_sparsity * z_sparsity
                + cfg.lambda_action_sparsity * a_sparsity
            )
            level_losses.append(pred_loss.detach())
            z_sparsity_achieved.append(z_act_frac)
            a_sparsity_achieved.append(a_act_frac)
            alpha_values.append(level.predictor.alpha.detach().clone())

            a_flat = out["a"].reshape(-1, out["a"].size(-1))
            a_norms = a_flat.norm(dim=-1)
            a_probs = torch.softmax(a_norms, dim=0)
            action_entropy = -(a_probs * a_probs.log().clamp(min=-100)).sum()
            action_entropies.append(action_entropy.detach())

            total_loss = total_loss + level_loss

        level_losses.reverse()
        z_sparsity_achieved.reverse()
        a_sparsity_achieved.reverse()
        alpha_values.reverse()
        action_entropies.reverse()

        return {
            "loss": total_loss,
            "level_losses": torch.stack(level_losses),
            "z": [out["z"] for out in level_outputs],
            "predicted": level_outputs[0]["predicted"],  # level-0 predicted features for autoregressive use
            "metrics": {
                "z_sparsity_achieved": torch.stack(z_sparsity_achieved),
                "a_sparsity_achieved": torch.stack(a_sparsity_achieved),
                "alpha": torch.stack(alpha_values),
                "action_entropy": torch.stack(action_entropies),
            },
        }

    def _sparsity_penalty(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the SDR sparsity penalty Ω(z) and actual activation fraction.

        Encourages a target activation fraction ρ::

            Ω(z) = (‖z‖₁ / d - ρ)²

        Parameters
        ----------
        z : torch.Tensor
            Latent activations, shape ``(B, d)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(penalty, mean_activation)`` — scalar penalty and actual activation fraction.
        """
        rho = self.config.sparsity_target
        d = z.size(-1)
        mean_activation = z.abs().sum(dim=-1).mean() / d
        return (mean_activation - rho) ** 2, mean_activation.detach()
