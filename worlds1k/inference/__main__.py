"""CLI entry point: ``uv run python -m worlds1k.inference``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from worlds1k.model.decoder import PixelDecoder
from worlds1k.model.encoders import build_frame_encoder
from worlds1k.model.frame_encoder import VideoEncoder
from worlds1k.model.world_model import WorldModel, WorldModelConfig

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="worlds1k.inference",
        description="Run prediction or dreaming with a trained world model.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="World model checkpoint.")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None, help="Pixel decoder checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Input video tensor (.pt file).")
    parser.add_argument("--mode", choices=["predict", "dream"], default="predict", help="Inference mode.")
    parser.add_argument("--dream-steps", type=int, default=64, help="Rollout steps in dream mode.")
    parser.add_argument("--output", type=Path, default=None, help="Output file for results (.pt).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info("device: %s", device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = WorldModelConfig()
    model = WorldModel.from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build encoder and load weights
    encoder = VideoEncoder(build_frame_encoder(config)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Optional decoder
    decoder = None
    if args.decoder_checkpoint is not None:
        decoder = PixelDecoder(config.d_latents[0], frame_height=config.image_size, frame_width=config.image_size)
        decoder.to(device)
        dec_ckpt = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
        decoder.load_state_dict(dec_ckpt["decoder"])
        decoder.eval()

    # Load input video tensor
    video = torch.load(args.input, map_location=device, weights_only=True)
    if video.dim() == 4:
        video = video.unsqueeze(0)  # add batch dim
    log.info("input: %s", video.shape)

    # Encode
    with torch.no_grad():
        features = encoder(video)

    if args.mode == "predict":
        from worlds1k.inference.predict import StatePredictor

        predictor = StatePredictor(model, decoder)
        result = predictor.predict(features)
        log.info("prediction z_last shape: %s", result["z_last"].shape)
        if "frames_pred" in result:
            log.info("predicted frame shape: %s", result["frames_pred"].shape)
    else:
        from worlds1k.inference.dream import Dreamer

        dreamer = Dreamer(model, decoder)
        result = dreamer.dream(features, num_steps=args.dream_steps)
        log.info("dream trajectory shape: %s", result["z_trajectory"].shape)
        if "frames_trajectory" in result:
            log.info("dreamed frames shape: %s", result["frames_trajectory"].shape)

    if args.output is not None:
        # Save tensors (convert to CPU for portability)
        save_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        torch.save(save_dict, args.output)
        log.info("saved to %s", args.output)


if __name__ == "__main__":
    main()
