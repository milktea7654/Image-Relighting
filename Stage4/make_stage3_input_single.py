from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--composite", required=True, type=Path)
    ap.add_argument("--render", required=True, type=Path)
    ap.add_argument("--albedo", required=True, type=Path)
    ap.add_argument("--roughness", required=True, type=Path)
    ap.add_argument("--metallic", required=True, type=Path)
    ap.add_argument("--glass-mask-soft", required=True, type=Path)
    ap.add_argument("--glass-mask", required=True, type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    args = ap.parse_args()

    row = {
        "scene_id": args.scene_id,
        "composite": str(args.composite),
        "render": str(args.render),
        "albedo": str(args.albedo),
        "roughness": str(args.roughness),
        "metallic": str(args.metallic),
        "glass_mask_soft": str(args.glass_mask_soft),
        "glass_mask": str(args.glass_mask),
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[DONE]", args.output_jsonl)


if __name__ == "__main__":
    main()
