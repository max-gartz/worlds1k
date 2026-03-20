#!/bin/bash
set -euo pipefail

# Cloud GPU setup script for worlds1k training.
# Run on a fresh Ubuntu instance (RunPod, Vast.ai, Lambda, etc.)
#
# Usage:
#   curl -sSL <raw-github-url>/scripts/cloud-setup.sh | bash
#   # or just copy-paste the commands below

echo "=== Installing dependencies ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg git > /dev/null 2>&1

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

echo "=== Cloning repo ==="
git clone https://github.com/max-gartz/worlds1k.git
cd worlds1k
uv venv --system-site-packages
uv sync

echo "=== Verifying setup ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"})')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')
"

nvidia-smi

uv run python -m worlds1k.training.pretrain --list-datasets

echo ""
echo "=== Ready! ==="
echo ""
echo "Set your HuggingFace token:"
echo "  export HF_TOKEN=hf_xxx"
echo ""
echo "Then run training:"
echo "  cd worlds1k"
echo "  uv run python -m worlds1k.training.pretrain \\"
echo "    --dataset disney --max-samples 500 --batch-size 8 \\"
echo "    --num-epochs 50 --eval-freq 50 --output-dir checkpoints/"
