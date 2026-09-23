#!/usr/bin/env bash
set -uo pipefail

export PATH="/root/.local/bin:${PATH}"
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export XDG_CACHE_HOME=/workspace/.cache
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/Stage3
mkdir -p logs runs /workspace/hf_cache/hub /workspace/hf_cache/datasets /workspace/tmp

backbones=(
  anyir
  xrestormer
  perceiveir
  instructir
  dcpt-promptir
)

for backbone in "${backbones[@]}"; do
  out_dir="runs/${backbone}"
  log_file="logs/${backbone}.log"
  echo "===== START ${backbone} $(date -Is) =====" | tee -a "${log_file}"
  if uv run python train_channel_ablation.py \
    --backbone "${backbone}" \
    --hf-dataset eenvil/stage3-relighting-dataset \
    --hf-cache-dir /workspace/hf_cache \
    --out-dir "${out_dir}" \
    --image-size 256 \
    --batch-size 1 \
    --epochs 30 \
    --lr 2e-4 \
    --num-workers 4 \
    --train-limit 1000 \
    --val-limit 512 \
    --eval-max-batches 512 \
    --log-every 500 \
    --use-albedo \
    --use-roughness \
    --use-metallic \
    --use-glass-mask 2>&1 | tee -a "${log_file}"; then
    echo "===== DONE ${backbone} $(date -Is) =====" | tee -a "${log_file}"
  else
    status=$?
    echo "===== FAILED ${backbone} status=${status} $(date -Is) =====" | tee -a "${log_file}"
  fi
done
