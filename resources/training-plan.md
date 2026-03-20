# Training Plan — Full-Scale Runs

Concrete plan for training Thousand Worlds beyond local smoke tests.

## Model Size Estimates

With default config (DINOv2-small backbone, 3 levels):

| Component | Parameters |
|-----------|-----------|
| DINOv2-small (frozen) | 22M (no gradients) |
| Projection layer | ~200K |
| Level 1 (Encoder + ActionHead + Predictor) | ~4M |
| Level 2 (Encoder + ActionHead + Predictor + TopDown) | ~2M |
| Level 3 (Encoder + ActionHead + Predictor + TopDown) | ~1M |
| **Total trainable** | **~7M** |
| FrameDecoder (phase 2) | ~2M |

With DINOv2-base backbone and wider latents (d_latents=[512, 256, 128]):

| Component | Parameters |
|-----------|-----------|
| DINOv2-base (frozen) | 86M (no gradients) |
| Projection + 3 levels | ~25M |
| **Total trainable** | **~25M** |

**Recommendation:** Start with default (~7M trainable). Scale to base (~25M) once training dynamics are validated.

## GPU Options (March 2026)

For this model size (~7M trainable), an **A100 is more than sufficient** and the best value. H100s are 1.5-2x faster but 2-3x more expensive per hour — not worth it at this scale.

| Provider | GPU | VRAM | $/GPU/hr | Notes |
|----------|-----|------|----------|-------|
| **Vast.ai (interruptible)** | A100 80GB | 80GB | ~$0.50 | Cheapest; needs checkpoint discipline |
| **Vast.ai (on-demand)** | A100 40GB | 40GB | ~$0.60 | Marketplace, variable availability |
| **Thunder Compute** | A100 80GB | 80GB | $0.78 | Budget reliable |
| **RunPod Community** | A100 80GB | 80GB | $0.89 | No minimum, per-second billing |
| **RunPod On-Demand** | A100 PCIe | 80GB | $1.19 | Most reliable |
| **Lambda Labs** | A100 SXM 40GB | 40GB | $1.48 | 2-week minimum — avoid for short runs |
| **RunPod Community** | H100 | 80GB | $1.99 | Only if you need speed |

**Recommendation:** RunPod Community A100 ($0.89/hr) for initial runs. Vast.ai interruptible ($0.50/hr) for budget experiments if you save checkpoints frequently. Avoid Lambda Labs unless committing to multi-week training.

## Training Configurations

### Config A: Validation Run (1–2 hours, ~$4)

Verify training works on cloud with real data. Single A100/H100.

```bash
uv run python -m thousand_worlds.train.world_model \
  --dataset disney \
  --max-samples 200 \
  --batch-size 4 \
  --num-epochs 10 \
  --eval-freq 20 \
  --output-dir checkpoints/val-run
```

- ~200 clips × 10 epochs = 2000 steps
- ~50 min on H100
- Cost: ~$2

### Config B: Small Scale (8–12 hours, ~$20)

Train on 1K clips for meaningful representation learning.

```bash
uv run python -m thousand_worlds.train.world_model \
  --dataset disney \
  --max-samples 1000 \
  --batch-size 8 \
  --num-epochs 30 \
  --eval-freq 50 \
  --learning-rate 3e-4 \
  --output-dir checkpoints/small
```

- 1000 clips × 30 epochs / batch 8 = ~3750 steps
- ~10 hours on H100
- Cost: ~$20

### Config C: Medium Scale (24–48 hours, ~$50–100)

Train on multiple datasets for robust features.

```bash
# Train on open-sora (high-res, diverse content)
uv run python -m thousand_worlds.train.world_model \
  --dataset open-sora \
  --max-samples 5000 \
  --batch-size 8 \
  --num-epochs 20 \
  --eval-freq 100 \
  --output-dir checkpoints/medium
```

- 5000 clips × 20 epochs / batch 8 = ~12500 steps
- ~36 hours on H100
- Cost: ~$70

### Config D: Large Scale with DINOv2-base (3–5 days, ~$200–400)

Wider model, more data, longer training. Requires accepting gated dataset TOS.

```bash
uv run python -m thousand_worlds.train.world_model \
  --dataset finevideo \
  --max-samples 20000 \
  --batch-size 8 \
  --encoder dinov2-base \
  --num-epochs 20 \
  --eval-freq 200 \
  --output-dir checkpoints/large
```

- 20000 clips × 20 epochs / batch 8 = 50K steps
- ~4 days on H100
- Cost: ~$200

## Phase 2: Decoder Training

After phase 1, train the frame decoder. This is lighter — the world model is frozen.

```bash
# Use same dataset, runs ~2-4x faster than phase 1
uv run python -c "
from thousand_worlds.train.decoder import DecoderTrainer, DecodeTrainConfig
from thousand_worlds.model.frame_decoder import FrameDecoder
# ... load phase 1 checkpoint, create decoder, train
"
```

Estimate: ~25% of phase 1 time (decoder is small, frozen encoder means less compute per step).

## Cloud Setup Script

```bash
# On a fresh GPU instance (Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/maximilian-gartz/thousand-worlds
cd thousand-worlds
uv sync

# Set HF token for gated datasets
export HF_TOKEN=hf_xxx

# Run training (use screen/tmux for persistence)
screen -S train
uv run python -m thousand_worlds.train.world_model \
  --dataset disney --max-samples 1000 --batch-size 8 \
  --num-epochs 30 --output-dir checkpoints/
```

## Estimated Total Budget

Using RunPod Community A100 at $0.89/hr:

| Phase | Config | Time | Cost |
|-------|--------|------|------|
| Validation | Config A | 1 hr | $1 |
| Small run | Config B | 10 hr | $9 |
| Medium run | Config C | 36 hr | $32 |
| Decoder (medium) | Phase 2 | 10 hr | $9 |
| **Total (recommended path)** | | **~57 hr** | **~$51** |

Using Vast.ai interruptible A100 at ~$0.50/hr, the total drops to ~$29.

For 10 hyperparameter experiments at Config B scale: ~$90 on RunPod, ~$50 on Vast.ai.

Start with Config A to validate cloud setup, then go to Config B or C for real training. Scale to Config D only after results look promising.
