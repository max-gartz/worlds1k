"""Unconstrained prediction rollout (dreaming) from an initial state.

Starting from encoded initial frames, autoregressively roll out the
world model's predictions without new observations.  The model
"dreams" forward in time using only its own predictions as input.

Run directly::

    uv run python -m worlds1k.inference.dream \\
        --checkpoint checkpoints/world_model.pt \\
        --decoder-checkpoint checkpoints/decoder.pt \\
        --dataset epic-kitchens --dream-steps 30 \\
        --output dream_demo.html
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from worlds1k.model.frame_decoder import FrameDecoder
    from worlds1k.model.world_model import WorldModel


class Dreamer:
    """Generate unconstrained rollouts from an initial state.

    Feeds the model's own predictions back as input to produce an
    extended imagined trajectory.
    """

    def __init__(
        self,
        model: WorldModel,
        decoder: FrameDecoder | None = None,
        audio_decoder: Any | None = None,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.audio_decoder = audio_decoder

    @torch.no_grad()
    def dream(self, seed_features: torch.Tensor, num_steps: int) -> dict[str, torch.Tensor]:
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
            ``"z_trajectory"`` — dreamed level-0 latents.
            ``"frames_trajectory"`` — dreamed pixel frames (if decoder).
        """
        self.model.eval()
        device = seed_features.device

        outputs = self.model(seed_features)
        z_level0 = outputs["z"][0]
        level0 = self.model.levels[0]

        z_curr = z_level0[:, -1, :]
        trajectory: list[torch.Tensor] = [z_curr]

        if z_level0.size(1) >= 2:
            action = level0.action_head(z_level0[:, -2, :], z_curr)
        else:
            action = torch.zeros(z_curr.size(0), level0.d_action, device=device)

        for _ in range(num_steps):
            z_next = level0.predictor(z_curr, action, context=None)
            trajectory.append(z_next)
            action = level0.action_head(z_curr, z_next)
            z_curr = z_next

        z_traj = torch.stack(trajectory, dim=1)
        result: dict[str, torch.Tensor] = {"z_trajectory": z_traj}

        if self.decoder is not None:
            b, t, d = z_traj.shape
            frames = self.decoder(z_traj.reshape(b * t, d))
            result["frames_trajectory"] = frames.view(b, t, *frames.shape[1:])

        if self.audio_decoder is not None:
            result["mel_trajectory"] = self.audio_decoder.decode_sequence(z_traj)

        return result


def _load_mp4(
    path: Path, window_size: int, image_size: int, *, with_audio: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Load an MP4 file and return (video, mel, raw_audio_waveform).

    video: (T, C, H, W) float [0, 1]
    mel: (T, 80, 3000) or None
    raw_audio: (samples,) float waveform or None
    """
    import torch.nn.functional as F  # noqa: N812
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    raw = path.read_bytes()
    vdec = VideoDecoder(raw)
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

    mel = None
    raw_audio = None

    if with_audio:
        try:
            adec = AudioDecoder(raw, sample_rate=16000)
            all_audio = adec.get_all_samples().data
            all_audio = all_audio.mean(dim=0) if all_audio.size(0) > 1 else all_audio.squeeze(0)

            # Extract raw audio for the seed window
            start_sample = int(start / fps * 16000)
            end_sample = int(end / fps * 16000)
            raw_audio = all_audio[start_sample:end_sample]

            # Compute per-frame mel spectrograms
            from transformers import WhisperFeatureExtractor

            fe = WhisperFeatureExtractor()
            mels = []
            half = 16000 * 15  # 15s half-window
            for i in range(window_size):
                center = int((start + i) / fps * 16000)
                a_start = max(0, center - half)
                a_end = a_start + 16000 * 30
                if a_end > all_audio.size(0):
                    a_end = all_audio.size(0)
                    a_start = max(0, a_end - 16000 * 30)
                chunk = all_audio[a_start:a_end]
                if chunk.size(0) < 16000 * 30:
                    chunk = F.pad(chunk, (0, 16000 * 30 - chunk.size(0)))
                m = fe(chunk.numpy(), sampling_rate=16000, return_tensors="pt")
                mels.append(m.input_features.squeeze(0))
            mel = torch.stack(mels)
        except Exception as e:
            print(f"no audio track: {e}")  # noqa: T201

    return video, mel, raw_audio


def _mel_to_waveform(mel: torch.Tensor, sample_rate: int = 16000, n_fft: int = 1024, n_mels: int = 80) -> torch.Tensor:
    """Convert mel spectrogram to waveform using Griffin-Lim.

    Parameters
    ----------
    mel : torch.Tensor
        Mel spectrogram, shape ``(n_mels, T_mel)``.

    Returns
    -------
    torch.Tensor
        Waveform, shape ``(samples,)``.
    """
    from torchaudio.transforms import GriffinLim, InverseMelScale

    inv_mel = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    griffin_lim = GriffinLim(n_fft=n_fft, hop_length=n_fft // 4)
    spec = inv_mel(mel)
    return griffin_lim(spec)


def _audio_to_wav_b64(audio: torch.Tensor, sample_rate: int = 16000) -> str:
    """Convert (samples,) float tensor to base64 WAV data URI."""
    import torchaudio

    buf = io.BytesIO()
    waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
    torchaudio.save(buf, waveform, sample_rate, format="wav")
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _frames_to_img_sequence(frames: torch.Tensor) -> list[str]:
    """Convert (T, C, H, W) float [0,1] to list of base64 PNG data URIs."""
    from PIL import Image

    imgs = []
    h, w = frames.shape[2], frames.shape[3]
    scale = max(1, 256 // h)
    for i in range(frames.size(0)):
        img = (frames[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        pil = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        imgs.append(f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}")
    return imgs


def _render_html(
    seed: torch.Tensor,
    dream: torch.Tensor,
    out: Path,
    dream_mel: torch.Tensor | None = None,
    seed_audio: torch.Tensor | None = None,
    seed_audio_sr: int = 16000,
) -> None:
    print("encoding seed frames...")  # noqa: T201
    seed_imgs = _frames_to_img_sequence(seed[:: max(1, seed.shape[0] // 8)])  # 8 evenly spaced
    seed_imgs_json = str(seed_imgs).replace("'", '"')

    seed_audio_tag = ""
    if seed_audio is not None:
        seed_wav = _audio_to_wav_b64(seed_audio, seed_audio_sr)
        seed_audio_tag = f'<audio controls src="{seed_wav}" style="margin-top:.5rem"></audio>'

    print("encoding dream frames...")  # noqa: T201
    dream_imgs = _frames_to_img_sequence(dream)
    dream_imgs_json = str(dream_imgs).replace("'", '"')

    dream_audio_tag = ""
    if dream_mel is not None:
        print("synthesizing audio from mel spectrogram (Griffin-Lim)...")  # noqa: T201
        dream_wav = _audio_to_wav_b64(_mel_to_waveform(dream_mel.squeeze(0)))
        dream_audio_tag = f'<audio controls src="{dream_wav}" style="margin-top:.5rem"></audio>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Thousand Worlds — Dream</title>
<style>
body {{ background:#0a0b0f; color:#e2e0dc; font-family:Georgia,serif; padding:2rem; max-width:900px; margin:0 auto }}
h1 {{ font-size:2rem; font-weight:normal }} h1 em {{ color:#c4a05c; font-style:italic }}
h2 {{ font-size:1rem; color:#c4a05c; font-family:monospace; margin:2rem 0 .5rem }}
p {{ color:#8a8880; font-size:.9rem }}
.player {{ border:1px solid #252630; border-radius:6px; display:inline-block; padding:4px; background:#000 }}
.player img {{ display:block; border-radius:4px }}
.row {{ margin:1rem 0 }}
.stats {{ font-family:monospace; font-size:.8rem; color:#5a5850; margin-top:2rem }}
.stats td {{ padding:.2rem 1rem .2rem 0 }}
</style></head><body>
<h1>Thousand <em>Worlds</em></h1>

<h2>Seed Video (ground truth)</h2>
<p>{seed.shape[0]} frames at {seed.shape[2]}x{seed.shape[3]}</p>
<div class="row">
  <div class="player"><img id="seed-img" width="384" height="384"></div>
  {seed_audio_tag}
</div>

<h2>Dream Sequence</h2>
<p>{dream.shape[0]} steps — autoregressive rollout, no sensory input after seed.</p>
<div class="row">
  <div class="player"><img id="dream-img" width="384" height="384"></div>
  {dream_audio_tag}
</div>

<table class="stats"><tr><td>seed frames</td><td>{seed.shape[0]}</td></tr>
<tr><td>dream steps</td><td>{dream.shape[0]}</td></tr>
<tr><td>resolution</td><td>{seed.shape[2]}x{seed.shape[3]}</td></tr></table>

<script>
function animate(imgId, frames, fps) {{
  const el = document.getElementById(imgId);
  let i = 0;
  el.src = frames[0];
  setInterval(() => {{ i = (i + 1) % frames.length; el.src = frames[i]; }}, 1000 / fps);
}}
const seedFrames = {seed_imgs_json};
const dreamFrames = {dream_imgs_json};
animate('seed-img', seedFrames, 8);
animate('dream-img', dreamFrames, 3);
</script>
</body></html>"""
    out.write_text(html)
    print(f"demo written to {out}")  # noqa: T201


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="worlds1k.inference.dream", description="Dream from a trained world model.")
    p.add_argument("--checkpoint", type=Path, required=True, help="World model checkpoint.")
    p.add_argument("--decoder-checkpoint", type=Path, default=None, help="Frame decoder checkpoint.")
    p.add_argument("--audio-decoder-checkpoint", type=Path, default=None, help="Audio decoder checkpoint.")
    p.add_argument("--input", type=Path, default=None, help="Input MP4 file (alternative to --dataset).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset for seed video.")
    p.add_argument("--max-videos", type=int, default=1)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--with-audio", action="store_true", help="Use AudioVideoEncoder for seed encoding.")
    p.add_argument("--dream-steps", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("dream.html"))
    args = p.parse_args(argv)

    import os

    from worlds1k.data import StreamingVideoDataset
    from worlds1k.model.frame_decoder import FrameDecoder
    from worlds1k.model.world_model import WorldModel, WorldModelConfig

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if args.with_audio:
        from worlds1k.model.audio_encoder import AudioVideoEncoder

        config = WorldModelConfig(image_size=args.image_size, d_input=512 + 256)
        model = WorldModel.from_config(config).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        encoder = AudioVideoEncoder.from_pretrained("dinov2-small", 512, "whisper-tiny", 256).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
    else:
        from worlds1k.model.encoder_base import build_frame_encoder
        from worlds1k.model.frame_encoder import VideoEncoder

        config = WorldModelConfig(image_size=args.image_size)
        model = WorldModel.from_config(config).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        encoder = VideoEncoder(build_frame_encoder(config)).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

    decoder = None
    if args.decoder_checkpoint:
        decoder = FrameDecoder(config.d_latents[0], frame_height=args.image_size, frame_width=args.image_size).to(
            device
        )
        dec_ckpt = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
        decoder.load_state_dict(dec_ckpt["decoder"])
        decoder.eval()

    audio_decoder = None
    if args.audio_decoder_checkpoint:
        from worlds1k.model.audio_decoder import AudioDecoder

        audio_decoder = AudioDecoder(config.d_latents[0]).to(device)
        adec_ckpt = torch.load(args.audio_decoder_checkpoint, map_location="cpu", weights_only=True)
        audio_decoder.load_state_dict(adec_ckpt["audio_decoder"])
        audio_decoder.eval()

    seed_video_raw_audio = None  # original audio waveform for the seed video

    if args.input is not None:
        seed_video, seed_mel, seed_video_raw_audio = _load_mp4(
            args.input, args.window_size, args.image_size, with_audio=args.with_audio
        )
        seed_video = seed_video.unsqueeze(0).to(device)
        print(f"seed: {seed_video.shape} (from {args.input})")  # noqa: T201
        with torch.no_grad():
            if args.with_audio and seed_mel is not None:
                features = encoder(seed_video, seed_mel.unsqueeze(0).to(device))
            else:
                features = encoder(seed_video)
    elif args.dataset is not None:
        from worlds1k.data import StreamingVideoDataset

        ds = StreamingVideoDataset(
            args.dataset,
            max_videos=args.max_videos,
            window_size=args.window_size,
            image_size=args.image_size,
            with_audio=args.with_audio,
            token=os.environ.get("HF_TOKEN"),
        )
        seed = next(iter(ds))
        seed_video = seed[0].unsqueeze(0).to(device)
        print(f"seed: {seed_video.shape}")  # noqa: T201
        with torch.no_grad():
            if args.with_audio and len(seed) > 1:
                features = encoder(seed_video, seed[1].unsqueeze(0).to(device))
            else:
                features = encoder(seed_video)
    else:
        print("error: provide --input or --dataset")  # noqa: T201
        return

    dreamer = Dreamer(model, decoder, audio_decoder)
    result = dreamer.dream(features, num_steps=args.dream_steps)
    print(f"dream: {result['z_trajectory'].shape}")  # noqa: T201

    if decoder and "frames_trajectory" in result:
        mel = result.get("mel_trajectory")
        mel_cpu = mel.cpu() if mel is not None else None
        _render_html(
            seed_video.squeeze(0).cpu(),
            result["frames_trajectory"].squeeze(0).cpu(),
            args.output,
            dream_mel=mel_cpu,
            seed_audio=seed_video_raw_audio,
        )
    else:
        torch.save(
            {k: v.cpu() for k, v in result.items() if isinstance(v, torch.Tensor)}, args.output.with_suffix(".pt")
        )
        print(f"saved latents to {args.output.with_suffix('.pt')}")  # noqa: T201


if __name__ == "__main__":
    main()
