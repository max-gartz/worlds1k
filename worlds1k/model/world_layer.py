"""Single hierarchical world layer of the predictive world model.

Each WorldLayer encodes a temporal window of lower-level states into a latent
representation, infers an action (transition code) between consecutive
latents, predicts the next latent given state + action + top-down context,
and projects its state downward as context for the layer below.

Core math (for level ℓ with temporal stride T_ℓ):

    Encode:   z_t^(ℓ) = E^(ℓ)(z_{t-T+1}^(ℓ-1), ..., z_t^(ℓ-1))
    Action:   a_t^(ℓ) = f_a^(ℓ)(z_t^(ℓ), z_{t+T}^(ℓ))
    Predict:  ẑ_{t+T}^(ℓ) = P^(ℓ)(z_t^(ℓ), a_t^(ℓ)) + α · c^(ℓ+1)
    Loss:     ‖z_{t+T}^(ℓ) - ẑ_{t+T}^(ℓ)‖² + λ_s Ω(z) + λ_a Ω(a)
"""

from __future__ import annotations

import torch
import torch.nn as nn


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

        # Input projection: d_input → d_model
        self.input_proj = nn.Linear(d_input, d_model)

        # Learned positional embeddings for the temporal window
        self.pos_emb = nn.Embedding(temporal_stride, d_model)

        # Causal transformer encoder
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

        # Final layer norm and readout
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, d_latent)

        # Register causal mask as buffer (True = blocked position)
        causal_mask = torch.triu(
            torch.ones(temporal_stride, temporal_stride, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, z_lower: torch.Tensor) -> torch.Tensor:
        """Encode a window of lower-level states into a single latent.

        Passes the temporal window through a causal transformer and
        reads out the final position to produce a single latent vector.

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

        # Project input and add positional embeddings
        positions = torch.arange(T, device=z_lower.device)
        x = self.input_proj(z_lower) + self.pos_emb(positions)  # (B, T, d_model)

        # Apply causal transformer
        x = self.transformer(x, mask=self.causal_mask[:T, :T])  # (B, T, d_model)

        # Read out from the last temporal position
        x = self.norm(x[:, -1, :])  # (B, d_model)
        return self.readout(x)  # (B, d_latent)


class ActionHead(nn.Module):
    """Infer the action (transition code) between consecutive latent states.

    The action a_t^(ℓ) captures *what changed* — it is not a motor command
    but a learned bottleneck that must be informative enough for prediction
    yet compact enough to generalize.  At level 1 this may learn optical
    flow; at higher levels it may learn abstract action concepts.

    Architecture: MLP with bottleneck.  Concatenates z_curr and z_next,
    passes through a hidden layer that narrows to d_action.

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
        d_hidden = d_latent  # bottleneck: 2*d_latent → d_latent → d_action
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
    """Predict the next latent state from current state, action, and top-down context.

    The predictor outputs a **delta/correction** that gets added to the
    top-down prior.  When top-down context is available::

        ẑ_{t+T}^(ℓ) = α · c^(ℓ+1) + P^(ℓ)(z_t^(ℓ), a_t^(ℓ))

    The top-down context is the baseline prediction; the predictor learns
    to correct it with local details.  For the highest level (no context),
    the predictor output is the full prediction.

    Architecture: 2-hidden-layer MLP with GELU activation.

    Parameters
    ----------
    d_latent : int
        Dimensionality of the latent state z^(ℓ).
    d_action : int
        Dimensionality of the action code a^(ℓ).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, d_latent: int, d_action: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_action = d_action
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
            nn.Linear(d_hidden, d_latent),
        )

    def forward(
        self,
        z_curr: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next latent state.

        Parameters
        ----------
        z_curr : torch.Tensor
            Current latent, shape ``(B, d_latent)``.
        action : torch.Tensor
            Inferred action code, shape ``(B, d_action)``.
        context : torch.Tensor or None
            Top-down context from level above, shape ``(B, d_latent)``.
            ``None`` for the highest level.

        Returns
        -------
        torch.Tensor
            Predicted next latent ẑ^(ℓ), shape ``(B, d_latent)``.
        """
        delta = self.net(torch.cat([z_curr, action], dim=-1))  # (B, d_latent)
        if context is not None:
            return self.alpha * context + delta
        return delta


class TopDownDecoder(nn.Module):
    """Project a higher-level latent down as context for the level below.

    Maps z^(ℓ) into the dimensionality of level ℓ-1 latents so it can
    serve as the baseline prediction in the lower-level predictor.

    Parameters
    ----------
    d_upper : int
        Dimensionality of this level's latent z^(ℓ).
    d_lower : int
        Dimensionality of the level below's latent z^(ℓ-1).
    """

    def __init__(self, d_upper: int, d_lower: int) -> None:
        super().__init__()
        self.d_upper = d_upper
        self.d_lower = d_lower
        self.proj = nn.Sequential(
            nn.Linear(d_upper, d_lower),
            nn.LayerNorm(d_lower),
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
            d_model=d_latent,  # use latent dim as transformer hidden dim
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.action_head = ActionHead(d_latent, d_action, dropout=dropout)
        self.predictor = Predictor(d_latent, d_action, dropout=dropout)
        self.top_down: TopDownDecoder | None = None
        if d_lower is not None:
            self.top_down = TopDownDecoder(d_latent, d_lower)

    def forward(
        self,
        z_lower_windows: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a full level forward pass.

        Parameters
        ----------
        z_lower_windows : torch.Tensor
            Consecutive temporal windows of lower-level states,
            shape ``(B, N_windows, T_ℓ, d_input)`` where ``N_windows >= 2``
            (need at least two consecutive latents for action inference).
        context : torch.Tensor or None
            Top-down context from the level above, shape ``(B, d_latent)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"z"`` — encoded latents, shape ``(B, N_windows, d_latent)``
            ``"a"`` — action codes, shape ``(B, N_windows - 1, d_action)``
            ``"z_pred"`` — predicted next latents, shape ``(B, N_windows - 1, d_latent)``
            ``"c_down"`` — top-down context for level below (if applicable)
        """
        B, N, T, D = z_lower_windows.shape

        # Encode each window into a latent vector
        z_flat = z_lower_windows.reshape(B * N, T, D)  # (B*N, T, d_input)
        z_enc = self.encoder(z_flat)  # (B*N, d_latent)
        z_enc = z_enc.view(B, N, -1)  # (B, N, d_latent)

        # Infer actions between consecutive latents
        z_curr = z_enc[:, :-1, :].reshape(B * (N - 1), -1)  # (B*(N-1), d_latent)
        z_next = z_enc[:, 1:, :].reshape(B * (N - 1), -1)  # (B*(N-1), d_latent)
        actions = self.action_head(z_curr, z_next)  # (B*(N-1), d_action)
        actions = actions.view(B, N - 1, -1)  # (B, N-1, d_action)

        # Predict next latents (with optional top-down context as baseline)
        ctx_expanded: torch.Tensor | None = None
        if context is not None:
            ctx_expanded = context.unsqueeze(1).expand(-1, N - 1, -1)
            ctx_expanded = ctx_expanded.reshape(B * (N - 1), -1)

        z_pred = self.predictor(
            z_curr,
            actions.view(B * (N - 1), -1),
            ctx_expanded,
        )
        z_pred = z_pred.view(B, N - 1, -1)  # (B, N-1, d_latent)

        result: dict[str, torch.Tensor] = {
            "z": z_enc,
            "a": actions,
            "z_pred": z_pred,
        }

        # Top-down context for the level below
        if self.top_down is not None:
            # Use the most recent latent as the top-down signal
            c_down = self.top_down(z_enc[:, -1, :])  # (B, d_lower)
            result["c_down"] = c_down

        return result
