# Training Plan

## Model Sizes

| Size | Trainable | Frozen | Backbone | Latents | Transformer | VRAM est. |
|------|-----------|--------|----------|---------|-------------|-----------|
| **Small** (current) | 5.6M | 22M | dinov2-small | [256, 128, 64] | 4L/8H | ~4 GB |
| **Medium** | 30M | 87M | dinov2-base | [512, 256, 128] | 6L/8H | ~12 GB |
| **Large** | 152M | 304M | dinov2-large | [1024, 512, 256] | 8L/16H | ~24 GB |
| **XL** | 870M | 304M | dinov2-large | [2048, 1024, 512] | 12L/16H | ~48 GB |

Small is for prototyping and validating the pipeline. Medium is where representation quality should start to matter. Large and XL are research-scale — comparable to V-JEPA 2's ~300M predictor.

CLI args for each size:

```bash
# Small (default)
--encoder dinov2-small

# Medium
--encoder dinov2-base

# Large (requires code change: d_input, d_latents, d_actions, transformer config)
# TODO: add --model-size flag or config file support

# XL
# Same — needs config file
```

Currently only Small and Medium are configurable via CLI (the `--encoder` flag sets the backbone). Large/XL require changing `WorldModelConfig` directly because latent dims and transformer depth aren't CLI args yet. Worth adding a `--model-size small|medium|large|xl` flag.

## GPU Requirements by Model Size

| Size | Min GPU | Recommended | $/hr |
|------|---------|-------------|------|
| Small (5.6M) | RTX 4090 (24 GB) | RTX 4090 | $0.35 |
| Medium (30M) | RTX 4090 (24 GB) | A100 40GB | $0.60 |
| Large (152M) | A100 40GB | A100 80GB | $0.89 |
| XL (870M) | A100 80GB | H100 80GB | $1.99 |

## Training Configurations

All configs use `--max-frames` as the training budget. The dataset yields infinitely from cached clips; training stops when the frame budget is reached.

### Phase 1: World Model

#### Quick Validation (~5 min, ~$0.03)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset ucf101 --max-frames 10000 --eval-freq 5
```

#### Small Model, Small Data (~30 min, ~$0.20)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset kinetics400-sample --max-frames 500000 \
  --batch-size 8 --eval-freq 50 \
  --output-dir checkpoints/small
```

#### Small Model, More Data (~3 hours, ~$1)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset disney --max-frames 5000000 \
  --batch-size 8 --eval-freq 100 \
  --output-dir checkpoints/small-long
```

#### Medium Model (~12 hours, ~$10)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset disney --max-frames 20000000 \
  --batch-size 8 --encoder dinov2-base \
  --eval-freq 200 \
  --output-dir checkpoints/medium
```

#### Audio+Video (EPIC-KITCHENS)

```bash
uv run python -m worlds1k.train.world_model \
  --dataset epic-kitchens --max-frames 5000000 \
  --batch-size 4 --with-audio --max-videos 10 \
  --output-dir checkpoints/epic-av
```

### Phase 2: Decoder Training

After phase 1, train vision + audio decoders. The world model is frozen — runs ~4x faster.

```bash
# Vision decoder only
uv run python -m worlds1k.train.decoder \
  --world-model checkpoints/latest.pt \
  --dataset disney --max-frames 1000000 \
  --output-dir checkpoints/decoders

# Vision + audio decoders
uv run python -m worlds1k.train.decoder \
  --world-model checkpoints/latest.pt \
  --dataset epic-kitchens --max-frames 1000000 \
  --with-audio --output-dir checkpoints/decoders
```

### Dreaming

```bash
# Video only
uv run python -m worlds1k.inference.dream \
  --world-model checkpoints/latest.pt \
  --vision-decoder checkpoints/decoders/vision_decoder.pt \
  --input video.mp4 --dream-steps 20 --output dream.html

# Video + audio
uv run python -m worlds1k.inference.dream \
  --world-model checkpoints/latest.pt \
  --vision-decoder checkpoints/decoders/vision_decoder.pt \
  --audio-decoder checkpoints/decoders/audio_decoder.pt \
  --input video.mp4 --dream-steps 20 --output dream.html
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
export WANDB_API_KEY=your_key  # optional

./scripts/train.sh disney 5000000
```

## Estimated Budgets

Using RunPod Community pricing:

| Scenario | Model | GPU | Frames | Time | Cost |
|----------|-------|-----|--------|------|------|
| Validate pipeline | Small | RTX 4090 | 50K | 10 min | $0.06 |
| First real run | Small | RTX 4090 | 5M | 3 hr | $1 |
| Meaningful features | Medium | A100 40GB | 20M | 12 hr | $7 |
| With audio | Small | A100 40GB | 5M | 6 hr | $4 |
| Research scale | Large | A100 80GB | 100M | 3 days | $64 |
| Full scale | XL | H100 | 500M | 10 days | $480 |

Start with Small to validate, scale model size and data together. There's no point training a Large model on 50K frames — the model capacity needs to match the data diversity.

## Scaling Principles

- **Match model size to data.** Small model (5.6M) saturates on ~5M frames. Medium (30M) needs ~20M+. Large (152M) needs ~100M+ diverse frames.
- **Increase resolution with model size.** Small: 64x64. Medium: 128x128. Large: 224x224. DINOv2 was trained on 224x224 so larger models benefit from native resolution.
- **Scale data diversity, not just quantity.** 100M frames from one video is worse than 10M frames from 1000 videos. Use multiple datasets.
- **Audio adds value at any scale.** Even Small + audio on EPIC-KITCHENS learns meaningful audiovisual correspondence.
