"""Unconstrained prediction rollout (dreaming) from an initial state.

Starting from encoded initial frames, autoregressively roll out the
world model's predictions without new observations.  The model
"dreams" forward in time using only its own predictions as input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from worlds1k.model.decoder import PixelDecoder
    from worlds1k.model.world_model import WorldModel


class Dreamer:
    """Generate unconstrained rollouts from an initial state.

    Feeds the model's own predictions back as input to produce an
    extended imagined trajectory.

    Parameters
    ----------
    model : WorldModel
        Trained hierarchical world model (should be in eval mode).
    decoder : PixelDecoder or None
        Optional pixel decoder for visualizing dreamed frames.
    """

    def __init__(self, model: WorldModel, decoder: PixelDecoder | None = None) -> None:
        self.model = model
        self.decoder = decoder

    @torch.no_grad()
    def dream(
        self,
        seed_features: torch.Tensor,
        num_steps: int,
    ) -> dict[str, torch.Tensor]:
        """Roll out predictions autoregressively from seed features.

        Parameters
        ----------
        seed_features : torch.Tensor
            Encoded seed frames, shape ``(B, T_seed, d_input)``.
        num_steps : int
            Number of forward prediction steps to unroll.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"z_trajectory"`` — dreamed level-0 latents, shape
            ``(B, num_steps, d_latent)``.
            ``"frames_trajectory"`` — dreamed pixel frames, shape
            ``(B, num_steps, C, H, W)`` (only if decoder is available).
        """
        self.model.eval()
        device = seed_features.device

        # Initial encoding — run the full model on seed features
        outputs = self.model(seed_features)
        z_level0 = outputs["z"][0]  # (B, N, d_latent)
        level0 = self.model.levels[0]

        # Start dreaming from the last encoded latent
        z_curr = z_level0[:, -1, :]  # (B, d_latent)
        trajectory: list[torch.Tensor] = [z_curr]

        # Infer the "action style" from the last transition to bootstrap
        if z_level0.size(1) >= 2:
            z_prev = z_level0[:, -2, :]
            action = level0.action_head(z_prev, z_curr)
        else:
            action = torch.zeros(z_curr.size(0), level0.d_action, device=device)

        for _ in range(num_steps):
            # Predict next latent using current state + action (no top-down context)
            z_next = level0.predictor(z_curr, action, context=None)
            trajectory.append(z_next)

            # Infer action for the next step
            action = level0.action_head(z_curr, z_next)
            z_curr = z_next

        z_traj = torch.stack(trajectory, dim=1)  # (B, num_steps+1, d_latent)

        result: dict[str, torch.Tensor] = {"z_trajectory": z_traj}

        if self.decoder is not None:
            b, t, d = z_traj.shape
            frames = self.decoder(z_traj.reshape(b * t, d))
            result["frames_trajectory"] = frames.view(b, t, *frames.shape[1:])

        return result
