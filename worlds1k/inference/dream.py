"""Unconstrained prediction rollout (dreaming) from an initial state.

Starting from encoded initial frames, autoregressively roll out the
world model's predictions without new observations.  The model
"dreams" forward in time using only its own predictions as input.

Run directly::

    uv run python -m worlds1k.inference.dream \\
        --world-model checkpoints/latest.pt \\
        --vision-decoder checkpoints/decoders/vision_decoder.pt \\
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
    from worlds1k.model.world_model import WorldModel


class Dreamer:
    """Generate unconstrained rollouts from an initial state.

    Feeds the model's own predictions back as input to produce an
    extended imagined trajectory.
    """

    def __init__(
        self,
        model: WorldModel,
        decoder: Any | None = None,
    ) -> None:
        self.model = model
        self.decoder = decoder

    @torch.no_grad()
    def dream(self, seed_features: torch.Tensor, num_steps: int) -> dict[str, torch.Tensor]:
        """Roll out predictions autoregressively with a sliding window.

        Runs the full hierarchical model at each step, shifting the
        feature window forward by one predicted frame.  The window size
        stays constant (same as seed length) so all hierarchy levels
        always have enough temporal context.

        Parameters
        ----------
        seed_features : torch.Tensor
            Encoded seed frames, shape ``(B, T_seed, d_input)``.
            T_seed must be large enough for all hierarchy levels.
        num_steps : int
            Number of forward prediction steps to unroll.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"z_trajectory"`` — dreamed level-0 latents.
            ``"frames_trajectory"`` — dreamed pixel frames (if decoder).
        """
        self.model.eval()

        features = seed_features
        output = self.model(features)
        predicted_features: list[torch.Tensor] = [features[:, -1, :]]  # last seed feature

        for _ in range(num_steps):
            next_feat = output["predicted"][:, -1:, :]  # (B, 1, d_input)
            features = torch.cat([features[:, 1:, :], next_feat], dim=1)  # slide window
            output = self.model(features)
            predicted_features.append(next_feat[:, 0, :])

        feat_traj = torch.stack(predicted_features, dim=1)  # (B, num_steps+1, d_input)
        result: dict[str, torch.Tensor] = {
            "z_trajectory": feat_traj,  # keep key name for compat
            "predicted_features": feat_traj,
        }

        if self.decoder is not None:
            b, t, d = feat_traj.shape
            frames = self.decoder.sample(feat_traj.reshape(b * t, d))
            result["frames_trajectory"] = frames.view(b, t, *frames.shape[1:])

        return result


def _load_mp4(
    path: Path, window_size: int, image_size: int,
) -> tuple[torch.Tensor, None, None]:
    """Load an MP4 file and return (video, None, None).

    video: (T, C, H, W) float [0, 1]
    """
    import torch.nn.functional as F  # noqa: N812
    from torchcodec.decoders import VideoDecoder

    raw = path.read_bytes()
    vdec = VideoDecoder(raw)
    n = vdec.metadata.num_frames

    start = torch.randint(0, max(1, n - window_size + 1), (1,)).item() if n >= window_size else 0
    end = min(start + window_size, n)
    video = vdec.get_frames_in_range(start=start, stop=end).data
    if video.size(0) < window_size:
        video = torch.cat([video, video[-1:].expand(window_size - video.size(0), -1, -1, -1)], dim=0)
    video = video.float() / 255.0
    if video.shape[-2] != image_size or video.shape[-1] != image_size:
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)

    return video, None, None


def _frames_to_mp4_b64(
    frames: torch.Tensor, fps: int = 10,
) -> str:
    """(T, C, H, W) float [0,1] -> base64 mp4 data URI."""
    import av
    from PIL import Image

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    v_stream = container.add_stream("h264", rate=fps)
    h, w = frames.shape[2], frames.shape[3]
    scale = max(1, 256 // h)
    v_stream.width = w * scale
    v_stream.height = h * scale
    v_stream.pix_fmt = "yuv420p"

    for i in range(frames.size(0)):
        img = (frames[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        pil = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
        frame = av.VideoFrame.from_image(pil)
        for packet in v_stream.encode(frame):
            container.mux(packet)
    for packet in v_stream.encode():
        container.mux(packet)

    container.close()
    return f"data:video/mp4;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _render_html(
    seed: torch.Tensor,
    dream: torch.Tensor,
    out: Path,
) -> None:
    print("encoding seed video...")  # noqa: T201
    seed_vid = _frames_to_mp4_b64(seed, fps=25)

    print("encoding dream video...")  # noqa: T201
    dream_vid = _frames_to_mp4_b64(dream, fps=5)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Thousand Worlds — Dream</title>
<style>
body {{ background:#0a0b0f; color:#e2e0dc; font-family:Georgia,serif; padding:2rem; max-width:900px; margin:0 auto }}
h1 {{ font-size:2rem; font-weight:normal }} h1 em {{ color:#c4a05c; font-style:italic }}
h2 {{ font-size:1rem; color:#c4a05c; font-family:monospace; margin:2rem 0 .5rem }}
p {{ color:#8a8880; font-size:.9rem }} video {{ border:1px solid #252630; border-radius:6px }}
.row {{ display:flex; gap:2rem; margin:1rem 0 }}
.stats {{ font-family:monospace; font-size:.8rem; color:#5a5850; margin-top:2rem }}
.stats td {{ padding:.2rem 1rem .2rem 0 }}
</style></head><body>
<h1>Thousand <em>Worlds</em></h1>
<h2>Seed Video (ground truth)</h2>
<p>{seed.shape[0]} frames at {seed.shape[2]}x{seed.shape[3]}</p>
<div class="row"><video width="384" height="384" controls loop>
<source src="{seed_vid}" type="video/mp4"></video></div>
<h2>Dream Sequence</h2>
<p>{dream.shape[0]} steps — autoregressive rollout, no sensory input after seed.</p>
<div class="row"><video width="384" height="384" controls loop>
<source src="{dream_vid}" type="video/mp4"></video></div>
<table class="stats"><tr><td>seed frames</td><td>{seed.shape[0]}</td></tr>
<tr><td>dream steps</td><td>{dream.shape[0]}</td></tr>
<tr><td>resolution</td><td>{seed.shape[2]}x{seed.shape[3]}</td></tr></table>
</body></html>"""
    out.write_text(html)
    print(f"demo written to {out}")  # noqa: T201


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="worlds1k.inference.dream", description="Dream from a trained world model.")
    p.add_argument("--world-model", type=Path, required=True, help="Path to world model checkpoint (.pt).")
    p.add_argument("--model", type=str, required=True, help="Named model config (e.g. 'world-3L-small').")
    p.add_argument("--vision-decoder", type=Path, default=None, help="Path to vision decoder checkpoint (.pt).")
    p.add_argument("--input", type=Path, default=None, help="Input MP4 file (alternative to --dataset).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset for seed video.")
    p.add_argument("--max-videos", type=int, default=1)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--dream-steps", type=int, default=20)
    p.add_argument("--num-inference-steps", type=int, default=20, help="DDIM steps for sampling (default: 20).")
    p.add_argument("--output", type=Path, default=Path("dream.html"))
    args = p.parse_args(argv)

    import os

    from worlds1k.model.configs import build_diffusion_decoder, get_config
    from worlds1k.model.world_model import WorldModel

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    config = get_config(args.model)
    config.image_size = args.image_size

    ckpt = torch.load(args.world_model, map_location="cpu", weights_only=True)

    from worlds1k.model.encoder_base import build_vision_encoder
    from worlds1k.model.vision_encoder import VideoEncoder

    model = WorldModel.from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    encoder = VideoEncoder(build_vision_encoder(config)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    decoder = None
    if args.vision_decoder:
        dec_ckpt = torch.load(args.vision_decoder, map_location="cpu", weights_only=True)
        arch = dec_ckpt.get("arch", "adagn")
        size = dec_ckpt.get("size", "base")
        num_inf = dec_ckpt.get("num_inference_steps", args.num_inference_steps)
        decoder = build_diffusion_decoder(config, arch=arch, size=size, num_inference_steps=num_inf).to(device)
        decoder.load_state_dict(dec_ckpt["decoder"])
        decoder.eval()

    if args.input is not None:
        seed_video, _, _ = _load_mp4(args.input, args.window_size, args.image_size)
        seed_video = seed_video.unsqueeze(0).to(device)
        print(f"seed: {seed_video.shape} (from {args.input})")  # noqa: T201
        with torch.no_grad():
            features = encoder(seed_video)
    elif args.dataset is not None:
        from worlds1k.data import StreamingVideoDataset

        ds = StreamingVideoDataset(
            args.dataset,
            max_videos=args.max_videos,
            window_size=args.window_size,
            image_size=args.image_size,
            token=os.environ.get("HF_TOKEN"),
        )
        seed = next(iter(ds))
        seed_video = seed[0].unsqueeze(0).to(device)
        print(f"seed: {seed_video.shape}")  # noqa: T201
        with torch.no_grad():
            features = encoder(seed_video)
    else:
        print("error: provide --input or --dataset")  # noqa: T201
        return

    dreamer = Dreamer(model, decoder)
    result = dreamer.dream(features, num_steps=args.dream_steps)
    print(f"dream: {result['z_trajectory'].shape}")  # noqa: T201

    if decoder and "frames_trajectory" in result:
        _render_html(
            seed_video.squeeze(0).cpu(),
            result["frames_trajectory"].squeeze(0).cpu(),
            args.output,
        )
    else:
        torch.save(
            {k: v.cpu() for k, v in result.items() if isinstance(v, torch.Tensor)}, args.output.with_suffix(".pt")
        )
        print(f"saved latents to {args.output.with_suffix('.pt')}")  # noqa: T201


if __name__ == "__main__":
    main()
