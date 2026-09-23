#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/workspace/runs/restormer"
ARCHIVE_DIR=""
INTERVAL=30
PYTHON_BIN="python"
ONCE=0

usage() {
  cat <<'EOF'
Usage:
  archive_checkpoints.sh [options]

Options:
  --run-dir PATH       Training run directory. Default: /workspace/runs/restormer
  --archive-dir PATH   Destination directory. Default: RUN_DIR/checkpoints
  --interval SECONDS   Poll interval. Default: 30
  --python PATH        Python executable with torch installed. Default: python
  --once               Archive currently available checkpoints once, then exit.
  -h, --help           Show this help.

This script watches checkpoint_last.pt and checkpoint_final.pt. Each readable
checkpoint is copied to ARCHIVE_DIR/checkpoint_epoch_XXXX.pt so checkpoint_last.pt
can keep being overwritten by training without losing older saved epochs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --archive-dir)
      ARCHIVE_DIR="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ARCHIVE_DIR" ]]; then
  ARCHIVE_DIR="$RUN_DIR/checkpoints"
fi

mkdir -p "$ARCHIVE_DIR"
MANIFEST="$ARCHIVE_DIR/manifest.tsv"

if [[ ! -f "$MANIFEST" ]]; then
  printf "archived_at\tsource\tepoch\tval_l1\tval_psnr\tdestination\n" > "$MANIFEST"
fi

wait_until_stable() {
  local path="$1"
  local previous=""
  local current=""

  for _ in $(seq 1 30); do
    [[ -f "$path" ]] || return 1
    current="$(stat -c '%s:%Y' "$path")"
    if [[ "$current" == "$previous" ]]; then
      return 0
    fi
    previous="$current"
    sleep 1
  done

  return 1
}

checkpoint_info() {
  local path="$1"
  CKPT_PATH="$path" "$PYTHON_BIN" - <<'PY'
import os
import torch

path = os.environ["CKPT_PATH"]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
epoch = int(ckpt.get("epoch", -1))
history = ckpt.get("history") or []
latest = history[-1] if history else {}
val_l1 = latest.get("val_l1", "na")
val_psnr = latest.get("val_psnr", "na")
print(f"{epoch}\t{val_l1}\t{val_psnr}")
PY
}

archive_checkpoint() {
  local source_name="$1"
  local source_path="$RUN_DIR/$source_name"

  [[ -f "$source_path" ]] || return 0

  if ! wait_until_stable "$source_path"; then
    echo "[WARN] $source_path is not stable/readable yet; will retry."
    return 0
  fi

  local info
  if ! info="$(checkpoint_info "$source_path")"; then
    echo "[WARN] Could not load $source_path yet; will retry."
    return 0
  fi

  local epoch val_l1 val_psnr
  IFS=$'\t' read -r epoch val_l1 val_psnr <<< "$info"

  local destination
  if [[ "$epoch" =~ ^-?[0-9]+$ ]] && [[ "$epoch" -ge 0 ]]; then
    destination="$ARCHIVE_DIR/checkpoint_epoch_$(printf '%04d' "$epoch").pt"
  else
    destination="$ARCHIVE_DIR/${source_name%.pt}_$(date +%Y%m%d_%H%M%S).pt"
  fi

  if [[ -f "$destination" ]]; then
    if cmp -s "$source_path" "$destination"; then
      return 0
    fi
    destination="${destination%.pt}_$(date +%Y%m%d_%H%M%S).pt"
  fi

  local tmp="${destination}.tmp.$$"
  cp -p "$source_path" "$tmp"
  mv -f "$tmp" "$destination"

  local archived_at
  archived_at="$(date -Is)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$archived_at" "$source_name" "$epoch" "$val_l1" "$val_psnr" "$destination" >> "$MANIFEST"

  echo "[OK] archived $source_name epoch=$epoch val_l1=$val_l1 val_psnr=$val_psnr -> $destination"
}

echo "[INFO] run dir    : $RUN_DIR"
echo "[INFO] archive dir: $ARCHIVE_DIR"
echo "[INFO] python     : $PYTHON_BIN"
echo "[INFO] interval   : ${INTERVAL}s"

while true; do
  archive_checkpoint "checkpoint_last.pt"
  archive_checkpoint "checkpoint_final.pt"

  if [[ "$ONCE" -eq 1 ]]; then
    break
  fi

  sleep "$INTERVAL"
done
