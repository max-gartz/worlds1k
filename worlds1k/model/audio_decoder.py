"""Audio decoder — maps latent states back to mel spectrograms.

Phase 2 module, trained with the world model frozen. Takes level-0
latent states and produces per-frame mel spectrograms that can be
concatenated and passed through a vocoder for waveform synthesis.

Architecture:
    z (B, d_latent) → Linear → reshape → ConvTranspose1d blocks → (B, n_mels, T_mel)
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_MELS = 80
T_MEL_PER_FRAME = 32  # ~200ms of audio per frame at standard Whisper mel params


class AudioDecoder(nn.Module):
    """Decode latent states to mel spectrograms.

    Each latent produces a short mel spectrogram chunk. During dreaming,
    chunks are concatenated along the time axis to form a continuous
    spectrogram, which can then be passed through a vocoder (HiFi-GAN,
    Vocos, etc.) for waveform synthesis.

    Parameters
    ----------
    d_latent : int
        Dimensionality of the input latent (level-0 by default).
    n_mels : int
        Number of mel frequency bins (default 80, matching Whisper).
    t_mel : int
        Mel time steps per frame (default 32, ~200ms of audio).
    """

    def __init__(self, d_latent: int, *, n_mels: int = N_MELS, t_mel: int = T_MEL_PER_FRAME) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_mels = n_mels
        self.t_mel = t_mel

        hidden = 256
        init_t = 4  # initial time dimension before upsampling

        self.proj = nn.Linear(d_latent, hidden * init_t)

        # Upsample: 4 → 8 → 16 → 32
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, n_mels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to mel spectrogram.

        Parameters
        ----------
        z : torch.Tensor
            Latent states, shape ``(B, d_latent)``.

        Returns
        -------
        torch.Tensor
            Mel spectrograms, shape ``(B, n_mels, t_mel)``.
        """
        x = self.proj(z)  # (B, hidden * init_t)
        x = x.view(x.size(0), 256, 4)  # (B, hidden, init_t)
        return self.decoder(x)  # (B, n_mels, t_mel)

    def decode_sequence(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of latents to a continuous mel spectrogram.

        Parameters
        ----------
        z_seq : torch.Tensor
            Latent sequence, shape ``(B, T, d_latent)``.

        Returns
        -------
        torch.Tensor
            Continuous mel spectrogram, shape ``(B, n_mels, T * t_mel)``.
        """
        b, t, d = z_seq.shape
        mels = self(z_seq.reshape(b * t, d))  # (B*T, n_mels, t_mel)
        mels = mels.view(b, t, self.n_mels, self.t_mel)
        return mels.permute(0, 2, 1, 3).reshape(b, self.n_mels, t * self.t_mel)
