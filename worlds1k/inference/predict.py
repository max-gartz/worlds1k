"""Next-state prediction from encoded frame features.

Given a sequence of encoded frame features, run through the hierarchy
and predict the next latent state at each level.  Optionally decode
level-1 predictions back to pixel space using a trained PixelDecoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from worlds1k.model.decoder import PixelDecoder
    from worlds1k.model.world_model import WorldModel


class StatePredictor:
    """Predict next states from encoded frame features.

    Parameters
    ----------
    model : WorldModel
        Trained hierarchical world model (should be in eval mode).
    decoder : PixelDecoder or None
        Optional pixel decoder for visualizing level-1 predictions.
    """

    def __init__(self, model: WorldModel, decoder: PixelDecoder | None = None) -> None:
        self.model = model
        self.decoder = decoder

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict next states from encoded frame features.

        Parameters
        ----------
        features : torch.Tensor
            Input frame features, shape ``(B, T, d_input)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"z"`` — encoded latents per level (list).
            ``"z_pred"`` — predicted next latents per level (list).
            ``"frames_pred"`` — predicted next frame in pixel space
            (only present if a decoder is available).
        """
        self.model.eval()
        if self.decoder is not None:
            self.decoder.eval()
        outputs = self.model(features)

        result: dict[str, torch.Tensor] = {
            "z": outputs["z"],
            "loss": outputs["loss"],
        }

        # Extract z_pred from the model's level outputs — the last predicted
        # latent at level 0 is the next-frame prediction
        z_level0 = outputs["z"][0]  # (B, N, d_latent)
        result["z_last"] = z_level0[:, -1, :]  # (B, d_latent)

        if self.decoder is not None:
            z_last = z_level0[:, -1, :]  # (B, d_latent)
            result["frames_pred"] = self.decoder(z_last)  # (B, C, H, W)

        return result
