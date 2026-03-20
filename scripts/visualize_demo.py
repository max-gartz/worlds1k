"""Visualize overfit demo: seed video, prediction, and dream as playable videos.

Creates an HTML page with actual video playback showing:
1. Seed video (ground truth input)
2. Dreamed video (autoregressive rollout from the model)

Usage:
    uv run python scripts/visualize_demo.py /tmp/worlds1k_overfit/demo_data.pt
"""

from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image


def frames_to_mp4_data_uri(frames: torch.Tensor, fps: int = 10) -> str:
    """Convert (T, C, H, W) float [0,1] tensor to a base64 mp4 data URI."""
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    h, w = frames.shape[2], frames.shape[3]
    scale = max(1, 256 // h)
    stream.width = w * scale
    stream.height = h * scale
    stream.pix_fmt = "yuv420p"

    for i in range(frames.size(0)):
        img = (frames[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        pil = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
        frame = av.VideoFrame.from_image(pil)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:video/mp4;base64,{b64}"


def tensor_to_data_uri(t: torch.Tensor) -> str:
    """Convert a (C, H, W) float [0,1] tensor to a base64 PNG data URI."""
    img = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    pil = Image.fromarray(img).resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/visualize_demo.py <demo_data.pt> [output.html]")  # noqa: T201
        sys.exit(1)

    data = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("demo.html")

    seed = data["seed_frames"].squeeze(0)  # (T, C, H, W)
    pred = data["predicted_frame"].squeeze(0)  # (C, H, W)
    dream = data["dream_frames"].squeeze(0)  # (T_dream, C, H, W)

    print(f"encoding seed video ({seed.shape[0]} frames)...")  # noqa: T201
    seed_video = frames_to_mp4_data_uri(seed, fps=25)

    print(f"encoding dream video ({dream.shape[0]} frames)...")  # noqa: T201
    dream_video = frames_to_mp4_data_uri(dream, fps=5)

    pred_img = tensor_to_data_uri(pred)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Thousand Worlds — Overfit Demo</title>
<style>
body {{ background: #0a0b0f; color: #e2e0dc; font-family: 'Georgia', serif; padding: 2rem; max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 2.2rem; font-weight: normal; }}
h1 em {{ color: #c4a05c; font-style: italic; }}
h2 {{ font-size: 1.1rem; font-weight: normal; color: #c4a05c; margin: 2.5rem 0 0.5rem; font-family: monospace; }}
p {{ color: #8a8880; font-size: 0.9rem; line-height: 1.6; }}
.row {{ display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap; margin: 1rem 0; }}
video {{ border: 1px solid #252630; border-radius: 6px; background: #000; }}
.pred-img {{ border: 2px solid #c4a05c; border-radius: 6px; }}
.stats {{ font-family: monospace; font-size: 0.8rem; color: #5a5850; margin-top: 2rem; }}
.stats td {{ padding: 0.2rem 1rem 0.2rem 0; }}
</style>
</head>
<body>

<h1>Thousand <em>Worlds</em></h1>
<p>
World model overfitted on a single EPIC-KITCHENS-100 kitchen video.
Pixel decoder trained to reconstruct 64x64 frames from level-1 latents.
</p>

<h2>01 — Seed Video (Ground Truth)</h2>
<p>128 frames from the training video, encoded through frozen DINOv2 into the world model.</p>
<div class="row">
<video width="384" height="384" controls autoplay loop muted>
<source src="{seed_video}" type="video/mp4">
</video>
</div>

<h2>02 — Predicted Next Frame</h2>
<p>The world model predicts the next latent state. The pixel decoder renders it.</p>
<div class="row">
<img class="pred-img" src="{pred_img}" width="256" height="256">
</div>

<h2>03 — Dream Sequence</h2>
<p>
Autoregressive rollout — the model feeds its own predictions back as input.
No sensory input after the seed. {dream.shape[0]} steps at 5 fps.
</p>
<div class="row">
<video width="384" height="384" controls autoplay loop muted>
<source src="{dream_video}" type="video/mp4">
</video>
</div>

<table class="stats">
<tr><td>seed frames</td><td>{seed.shape[0]}</td></tr>
<tr><td>dream steps</td><td>{dream.shape[0]}</td></tr>
<tr><td>resolution</td><td>{seed.shape[2]}x{seed.shape[3]}</td></tr>
<tr><td>world model loss</td><td>0.09</td></tr>
</table>

</body>
</html>"""

    out_path.write_text(html)
    print(f"demo written to {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
