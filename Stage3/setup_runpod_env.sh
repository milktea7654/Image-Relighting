#!/usr/bin/env bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1 || ! command -v grep >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git grep curl ca-certificates
fi

export PATH="/root/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  sh /tmp/uv-install.sh
  export PATH="/root/.local/bin:${PATH}"
fi

mkdir -p /workspace/hf_cache/hub /workspace/hf_cache/datasets /workspace/.cache/uv /workspace/.cache/pip /workspace/tmp /root/.cache
if [ -e /root/.cache/huggingface ] && [ ! -L /root/.cache/huggingface ]; then
  mv /root/.cache/huggingface /workspace/hf_cache_root_backup
fi
if [ -e /root/.cache/uv ] && [ ! -L /root/.cache/uv ]; then
  mv /root/.cache/uv /workspace/uv_cache_root_backup
fi
if [ -e /root/.cache/pip ] && [ ! -L /root/.cache/pip ]; then
  mv /root/.cache/pip /workspace/pip_cache_root_backup
fi
ln -sfn /workspace/hf_cache /root/.cache/huggingface
ln -sfn /workspace/.cache/uv /root/.cache/uv
ln -sfn /workspace/.cache/pip /root/.cache/pip

export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export XDG_CACHE_HOME=/workspace/.cache
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/tmp

cd /workspace/Stage3
uv sync --no-dev
uv pip install thop lightning pytorch-lightning fvcore torchstat
uv run python -m py_compile \
  stage3_anyir.py \
  stage3_xrestormer.py \
  stage3_perceiveir.py \
  stage3_instructir.py \
  stage3_dcpt.py \
  stage3_unet.py \
  train_channel_ablation.py
uv run python train_channel_ablation.py --help | grep -E "anyir|xrestormer|perceiveir|instructir|dcpt-promptir"
echo INSTALL_READY
