# Training Plan — Full-Scale Runs

Concrete plan for training worlds1k beyond local smoke tests.

## Model Size Estimates

With default config (DINOv2-small backbone, 3 levels):

| Component | Parameters |
|-----------|-----------|
| DINOv2-small (frozen) | 22M (no gradients) |
| Projection layer | ~200K |
| Level 1 (Encoder + ActionHead + Predictor) | ~4M |
| Level 2 (Encoder + ActionHead + Predictor + TopDown) | ~2M |
| Level 3 (Encoder + ActionHead + Predictor + TopDown) | ~1M |
| **Total trainable** | **~5.6M** |
| FrameDecoder (phase 2) | ~2M |
| AudioDecoder (phase 2) | ~0.5M |

## GPU Options (March 2026)

For ~5.6M trainable params, even an RTX 4090 or L4 (24GB) is sufficient. A100 is overkill but cheap on community clouds.

| Provider | GPU | VRAM | $/GPU/hr | Notes |
|----------|-----|------|----------|-------|
| **Vast.ai (interruptible)** | A100 80GB | 80GB | ~$0.50 | Cheapest; needs checkpoint discipline |
| **Thunder Compute** | A100 80GB | 80GB | $0.78 | Budget reliable |
| **RunPod Community** | A100 80GB | 80GB | $0.89 | No minimum, per-second billing |
| **RunPod Community** | RTX 4090 | 24GB | $0.35 | Sufficient for this model size |

## Training Configurations

All configs use `--max-frames` as the single training duration knob. The dataset yields infinitely from cached clips; training stops when the frame budget is reached.

### Config A: Validation Run (~$0.15)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset disney --max-frames 50000 \
  --batch-size 4 --eval-freq 20 \
  --output-dir checkpoints/val-run
```

- 50K frames / (4 x 128) = ~98 steps, ~10 min

### Config B: Small Scale (~$2)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset kinetics400-sample --max-frames 500000 \
  --batch-size 8 --eval-freq 50 \
  --output-dir checkpoints/small
```

- 500K frames / (8 x 128) = ~488 steps, ~2 hours

### Config C: Medium Scale (~$10)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset disney --max-frames 5000000 \
  --batch-size 8 --eval-freq 100 \
  --output-dir checkpoints/medium
```

- 5M frames, ~12 hours on A100

### Config D: Audio+Video (EPIC-KITCHENS)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset epic-kitchens --max-frames 5000000 \
  --batch-size 4 --with-audio --max-videos 10 \
  --output-dir checkpoints/epic-av
```

## Phase 2: Decoder Training

After phase 1, train frame + audio decoders. The world model is frozen.

```bash
# Frame decoder only
uv run python -m worlds1k.train.decoder \
  --checkpoint checkpoints/latest.pt \
  --dataset disney --max-frames 500000 \
  --output-dir checkpoints/decoders

# Frame + audio decoders
uv run python -m worlds1k.train.decoder \
  --checkpoint checkpoints/latest.pt \
  --dataset epic-kitchens --max-frames 500000 \
  --with-audio --output-dir checkpoints/decoders
```

## Dreaming

```bash
# Video only
uv run python -m worlds1k.inference.dream \
  --checkpoint checkpoints/latest.pt \
  --decoder-checkpoint checkpoints/decoders/frame_decoder.pt \
  --input video.mp4 --dream-steps 20

# Video + audio (original audio on seed, Griffin-Lim on dream)
uv run python -m worlds1k.inference.dream \
  --checkpoint checkpoints/latest.pt \
  --decoder-checkpoint checkpoints/decoders/frame_decoder.pt \
  --audio-decoder-checkpoint checkpoints/decoders/audio_decoder.pt \
  --input video.mp4 --dream-steps 20
```

## Cloud Setup

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/max-gartz/worlds1k
cd worlds1k
uv venv --system-site-packages
uv sync

export HF_TOKEN=hf_xxx
export WANDB_API_KEY=your_key  # optional, enables wandb logging

./scripts/train.sh disney 500000
```

## Estimated Total Budget

Using RunPod Community A100 at $0.89/hr:

| Phase | Config | Time | Cost |
|-------|--------|------|------|
| Validation | Config A | 10 min | $0.15 |
| Small run | Config B | 2 hr | $2 |
| Medium run | Config C | 12 hr | $11 |
| Decoders | Phase 2 | 3 hr | $3 |
| **Total (recommended path)** | | **~17 hr** | **~$16** |

Start with Config A to validate cloud setup, then scale up.
