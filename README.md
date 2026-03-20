# Thousand Worlds

A hierarchical predictive world model that learns multi-scale temporal dynamics from video.

Rooted in Jeff Hawkins' theory of cortical prediction hierarchies — the brain builds a model of the world by predicting sensory input at multiple timescales, with higher cortical levels sending top-down predictions that constrain lower levels. This architecture implements that principle: a three-level hierarchy where each level encodes, predicts, and learns from prediction error, entirely in latent space.

**[Project Page](https://max-gartz.github.io/worlds1k)** · **[Architecture](#architecture)** · **[Quick Start](#quick-start)**

## Key Ideas

- **Predict in latent space.** A frozen DINOv2 backbone encodes frames. The hierarchy predicts future *representations*, not pixels. No reconstruction loss during training.
- **Temporal hierarchy.** Level 1 sees every frame (~33ms). Level 2 compresses 8 frames (~250ms). Level 3 compresses 64 frames (~2s). Top-down context flows from abstract to concrete.
- **Emergent action codes.** Transition codes between states are learned as a prediction bottleneck, not from labels. Different levels discover different abstractions: optical flow, object actions, scene transitions.
- **Sparse distributed representations.** Sparsity penalties prevent collapse and encourage disentangled codes with exponential representational capacity.

## Architecture

```
Frame x_t
  │
  ▼
E¹: encode frame → z_t^(1)              [every frame, ~33ms]
  │
  ├─→ P¹: predict z_{t+1}^(1) from z_t, a_t, context
  │
  ▼ (every 8 frames)
E²: encode 8 z^(1)'s → z_k^(2)          [~250ms]
  │
  ├─→ P²: predict z_{k+1}^(2)
  ├─→ D²: top-down context → level 1
  │
  ▼ (every 64 frames)
E³: encode 8 z^(2)'s → z_m^(3)           [~2s]
  │
  ├─→ P³: predict z_{m+1}^(3)
  ├─→ D³: top-down context → level 2
```

Training objective:

```
L = Σ_l γ_l E_t[||z - ẑ||²] + λ_s Σ_l E_t[Ω(z)] + λ_a Σ_l E_t[Ω(a)]
```

## Quick Start

Requires Python 3.11+, [FFmpeg](https://ffmpeg.org/), and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/max-gartz/worlds1k
cd worlds1k
uv sync
```

### Train

```bash
# List available datasets
uv run python -m worlds1k.train.world_model --list-datasets

# Smoke test
uv run python -m worlds1k.train.world_model \
  --dataset ucf101 --max-frames 5000 --eval-freq 5

# Real training
HF_TOKEN=hf_xxx uv run python -m worlds1k.train.world_model \
  --dataset disney --max-frames 500000 --output-dir checkpoints/

# Train with audio (EPIC-KITCHENS)
HF_TOKEN=hf_xxx uv run python -m worlds1k.train.world_model \
  --dataset epic-kitchens --max-frames 100000 --with-audio --max-videos 1
```

### Decode (Phase 2)

```bash
uv run python -m worlds1k.train.decoder \
  --checkpoint checkpoints/latest.pt --dataset disney --max-frames 50000 \
  --output-dir checkpoints/decoders

# With audio decoder
uv run python -m worlds1k.train.decoder \
  --checkpoint checkpoints/latest.pt --dataset epic-kitchens --max-frames 50000 \
  --with-audio --output-dir checkpoints/decoders
```

### Dream

```bash
# Video only
uv run python -m worlds1k.inference.dream \
  --checkpoint checkpoints/latest.pt \
  --decoder-checkpoint checkpoints/decoders/frame_decoder.pt \
  --input video.mp4 --dream-steps 20

# Video + audio
uv run python -m worlds1k.inference.dream \
  --checkpoint checkpoints/latest.pt \
  --decoder-checkpoint checkpoints/decoders/frame_decoder.pt \
  --audio-decoder-checkpoint checkpoints/decoders/audio_decoder.pt \
  --input video.mp4 --dream-steps 20
```

### Test

```bash
uv run pytest                    # unit tests (~5s)
uv run pytest -m integration    # + streaming integration tests
```

## Datasets

Streams directly from HuggingFace — no full downloads needed.

| Name | Path | Status |
|------|------|--------|
| `ucf101` | sayakpaul/ucf101-subset | Open |
| `disney` | Wild-Heart/Disney-VideoGeneration-Dataset | Open |
| `open-sora` | LanguageBind/Open-Sora-Plan-v1.1.0 | Open |
| `kinetics400-sample` | JackWong0911/kinetic-400_450samples | Open |
| `epic-kitchens` | awsaf49/epic_kitchens_100 | Open (file-based, 501 GB) |
| `finevideo` | HuggingFaceFV/finevideo | Gated |
| `egocentric-10k` | builddotai/Egocentric-10K | Gated |

Any HuggingFace dataset with a `Video` feature works: `--dataset org/dataset-name`.

## Project Structure

```
worlds1k/
  model/
    world_model.py      # Hierarchical predictive model
    world_layer.py      # Single hierarchy level
    encoder_base.py     # Abstract base classes + factories
    frame_encoder.py    # DINOv2 visual encoder
    audio_encoder.py    # Whisper audio encoder + AudioVideoEncoder
    frame_decoder.py    # Frame decoder (phase 2)
    audio_decoder.py    # Audio decoder (phase 2, mel spectrograms)
  train/
    world_model.py      # Phase 1: world model training + CLI
    decoder.py          # Phase 2: frame + audio decoder training + CLI
  inference/
    dream.py            # Dreaming (autoregressive rollout) + CLI
  data.py               # Dataset registry + streaming with disk cache
```

## Background

This work is grounded in Jeff Hawkins' *Thousand Brains Theory* — the idea that the neocortex consists of many parallel predictive models, each operating at different timescales and levels of abstraction. Predictions flow down the cortical hierarchy; prediction errors flow up. Learning is driven entirely by the discrepancy between what was predicted and what was observed.

The architecture shares principles with LeCun's JEPA (Joint Embedding Predictive Architecture), particularly the commitment to latent-space prediction over pixel reconstruction. Where V-JEPA operates at a single temporal scale, this project implements the multi-scale temporal hierarchy that both Hawkins and LeCun theorized but neither has fully built.

## License

MIT
