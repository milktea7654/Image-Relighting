from __future__ import annotations

import gc

import torch

from stage3_unet import build_stage3_model, count_parameters


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ["anyir", "xrestormer", "perceiveir", "instructir", "dcpt-promptir"]
    print("device", device, flush=True)
    for name in names:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = build_stage3_model(name, input_channels=12).to(device)
        x = torch.rand(1, 12, 256, 256, device=device)
        composite = x[:, :3]
        y = model(x, composite)
        loss = y.mean()
        loss.backward()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        print(
            f"{name}: params={count_parameters(model)} shape={tuple(y.shape)} "
            f"finite={torch.isfinite(y).all().item()} peak_gb={peak_gb:.2f}",
            flush=True,
        )
        del model, x, composite, y, loss
        gc.collect()


if __name__ == "__main__":
    main()
