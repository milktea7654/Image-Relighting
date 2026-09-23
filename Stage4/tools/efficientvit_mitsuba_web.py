from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import online_efficientvit_pipeline as core


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EfficientViT Mitsuba Render</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101315;
      --panel: #181d20;
      --line: #334047;
      --text: #eef3f4;
      --muted: #9aa8ad;
      --accent: #66d2b3;
      --warn: #f0bd61;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--text); font: 14px/1.35 system-ui, "Segoe UI", sans-serif; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 370px; min-height: 100vh; }
    .viewer { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; min-width: 0; background: #090b0c; }
    .bar { min-height: 42px; display: flex; align-items: center; gap: 12px; padding: 8px 14px; border-bottom: 1px solid var(--line); color: var(--muted); }
    .bar strong { color: var(--text); }
    .images { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; min-height: 0; background: var(--line); }
    figure { margin: 0; min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); background: #090b0c; }
    figcaption { padding: 8px 10px; color: var(--muted); border-bottom: 1px solid var(--line); }
    img { width: 100%; height: 100%; object-fit: contain; min-height: 0; background: #050606; }
    aside { border-left: 1px solid var(--line); background: var(--panel); padding: 14px; overflow: auto; }
    h1 { margin: 0 0 12px; font-size: 18px; }
    h2 { margin: 20px 0 10px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }
    label { display: grid; grid-template-columns: 92px minmax(0, 1fr) 66px; gap: 9px; align-items: center; margin: 9px 0; color: var(--muted); }
    input { width: 100%; }
    input[type=range] { accent-color: var(--accent); }
    input[type=number] { padding: 5px 6px; color: var(--text); background: #20272b; border: 1px solid var(--line); border-radius: 6px; }
    button { width: 100%; min-height: 40px; margin-top: 12px; border: 1px solid #397d6e; border-radius: 7px; background: #1f3b35; color: var(--text); font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    pre { min-height: 150px; padding: 10px; overflow: auto; background: #0b0e0f; border: 1px solid var(--line); border-radius: 7px; color: var(--muted); font: 12px/1.4 Consolas, monospace; white-space: pre-wrap; }
    .footer { border-top: 1px solid var(--line); border-bottom: 0; justify-content: space-between; }
    .pill { color: #101315; background: var(--warn); padding: 2px 7px; border-radius: 999px; font-weight: 700; }
    @media (max-width: 1050px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid var(--line); } .images { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section class="viewer">
      <div class="bar"><strong>Persistent Mitsuba + EfficientViT</strong><span id="status">ready</span></div>
      <div class="images">
        <figure><figcaption>render_in</figcaption><img id="renderImg"></figure>
        <figure><figcaption>composite_in</figcaption><img id="compositeImg"></figure>
        <figure><figcaption>efficientvit_prediction</figcaption><img id="predImg"></figure>
      </div>
      <div class="bar footer"><span id="lastTiming">no render yet</span><span class="pill">button-rendered</span></div>
    </section>
    <aside>
      <h1>Light Controls</h1>
      <h2>Environment</h2>
      <label>Env R <input id="envR" type="range" min="0" max="2" step="0.01" value="0.2"><input data-for="envR" type="number" step="0.01"></label>
      <label>Env G <input id="envG" type="range" min="0" max="2" step="0.01" value="0.2"><input data-for="envG" type="number" step="0.01"></label>
      <label>Env B <input id="envB" type="range" min="0" max="2" step="0.01" value="0.2"><input data-for="envB" type="number" step="0.01"></label>
      <h2>Point Light</h2>
      <label>X <input id="x" type="range" min="-10" max="10" step="0.01" value="-4.47"><input data-for="x" type="number" step="0.01"></label>
      <label>Y <input id="y" type="range" min="-8" max="8" step="0.01" value="-0.65"><input data-for="y" type="number" step="0.01"></label>
      <label>Z <input id="z" type="range" min="-40" max="-0.05" step="0.01" value="-20"><input data-for="z" type="number" step="0.01"></label>
      <label>Red <input id="r" type="range" min="0" max="160" step="0.5" value="35"><input data-for="r" type="number" step="0.5"></label>
      <label>Green <input id="g" type="range" min="0" max="160" step="0.5" value="32"><input data-for="g" type="number" step="0.5"></label>
      <label>Blue <input id="b" type="range" min="0" max="160" step="0.5" value="26"><input data-for="b" type="number" step="0.5"></label>
      <h2>Render</h2>
      <label>SPP <input id="spp" type="range" min="1" max="64" step="1" value="1"><input data-for="spp" type="number" step="1"></label>
      <button id="renderBtn">Render Scene</button>
      <h2>Timing</h2>
      <pre id="metrics">{}</pre>
    </aside>
  </main>
  <script>
    const ids = ["envR", "envG", "envB", "x", "y", "z", "r", "g", "b", "spp"];
    const controls = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
    for (const input of document.querySelectorAll("input[type=number][data-for]")) input.value = controls[input.dataset.for].value;
    for (const id of ids) controls[id].addEventListener("input", () => syncFrom(id));
    for (const input of document.querySelectorAll("input[type=number][data-for]")) input.addEventListener("input", () => {
      controls[input.dataset.for].value = input.value;
    });
    function syncFrom(id) {
      const n = document.querySelector(`input[type=number][data-for="${id}"]`);
      if (n) n.value = controls[id].value;
    }
    function v(id) { return Number(controls[id].value); }
    async function render() {
      const btn = document.getElementById("renderBtn");
      btn.disabled = true;
      document.getElementById("status").textContent = "rendering";
      const payload = {
        env_rgb: [v("envR"), v("envG"), v("envB")],
        point_light_position: [v("x"), v("y"), v("z")],
        point_light_rgb: [v("r"), v("g"), v("b")],
        spp: Math.max(1, Math.round(v("spp")))
      };
      try {
        const res = await fetch("/api/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        const t = Date.now();
        document.getElementById("renderImg").src = data.render_url + "?t=" + t;
        document.getElementById("compositeImg").src = data.composite_url + "?t=" + t;
        document.getElementById("predImg").src = data.prediction_url + "?t=" + t;
        document.getElementById("metrics").textContent = JSON.stringify(data.timing, null, 2);
        document.getElementById("lastTiming").textContent = `total ${(data.timing.total_ms).toFixed(1)} ms | mitsuba ${(data.timing.mitsuba_ms).toFixed(1)} ms | model ${(data.timing.model_ms).toFixed(1)} ms`;
        document.getElementById("status").textContent = "ready";
      } catch (err) {
        document.getElementById("status").textContent = "error";
        document.getElementById("metrics").textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    }
    document.getElementById("renderBtn").onclick = render;
    fetch("/api/status").then(r => r.json()).then(d => {
      document.getElementById("metrics").textContent = JSON.stringify(d, null, 2);
    });
  </script>
</body>
</html>"""


def np_to_chw_tensor(arr: np.ndarray, image_size: int, gray: bool = False) -> torch.Tensor:
    arr = np.clip(arr, 0.0, 1.0)
    if gray:
        if arr.ndim == 3:
            arr = arr[..., :1]
        img = Image.fromarray((arr[..., 0] * 255.0).round().astype(np.uint8), "L")
        img = img.resize((image_size, image_size), Image.BICUBIC)
        out = np.asarray(img, dtype=np.float32)[None, ...] / 255.0
        return torch.from_numpy(out).contiguous()
    img = Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


class PersistentEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scene_dir = args.scene_dir.resolve()
        self.out_dir = args.out_dir.resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seed = int(args.seed)

        t0 = time.perf_counter()
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        self.model, self.input_names, self.channels = core.load_efficientvit(args.checkpoint.resolve(), args.stage3_root.resolve(), self.device)
        self._preload_static_tensors()
        self._load_mitsuba_scene()
        self._warmup_model()
        self.startup_seconds = time.perf_counter() - t0

    def _preload_static_tensors(self) -> None:
        self.static_tensors: dict[str, torch.Tensor] = {}
        paths = {
            "albedo": self.scene_dir / "albedo.png",
            "roughness": self.scene_dir / "roughness.png",
            "metallic": self.scene_dir / "metallic.png",
            "normal": self.scene_dir / "normal.png",
            "shading": self.scene_dir / "irradiance.png",
        }
        for name, path in paths.items():
            if name in self.input_names and path.exists():
                loader = core.gray_tensor if name in core.GRAY_INPUTS else core.rgb_tensor
                self.static_tensors[name] = loader(path, self.args.image_size)
        if "glass_mask" in self.input_names:
            if self.args.black_glass_mask:
                self.static_tensors["glass_mask"] = torch.zeros(1, self.args.image_size, self.args.image_size, dtype=torch.float32)
            elif self.args.glass_mask and self.args.glass_mask.exists():
                self.static_tensors["glass_mask"] = core.gray_tensor(self.args.glass_mask, self.args.image_size)

    def _load_mitsuba_scene(self) -> None:
        import mitsuba as mi

        self.mi = mi
        mi.set_variant(self.args.mitsuba_variant)
        scene_meta = json.loads((self.scene_dir / "scene_mitsuba.json").read_text(encoding="utf-8"))
        mesh_path = core.find_render_backend(self.scene_dir, scene_meta)
        scene_args = SimpleNamespace(spp=self.args.spp, render_width=self.args.render_width, render_height=self.args.render_height)
        self.scene_dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 4},
            "sensor": core.make_sensor(mi, scene_meta, self.args.spp, self.args.render_width, self.args.render_height),
            "shape": {
                "type": "obj" if mesh_path.suffix.lower() == ".obj" else "ply",
                "filename": str(mesh_path),
                "bsdf": core.make_bsdf(self.scene_dir, scene_meta),
            },
            "env_light": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.2, 0.2, 0.2]},
            },
            "target_point": {
                "type": "point",
                "position": [-4.47, -0.65, -20.0],
                "intensity": {"type": "rgb", "value": [35.0, 32.0, 26.0]},
            },
        }
        self.scene = mi.load_dict(self.scene_dict)
        self.params = mi.traverse(self.scene)
        self.param_keys = list(self.params.keys())
        self.env_key = self._find_key(["env_light", "radiance", "value"])
        self.point_rgb_key = self._find_key(["target_point", "intensity", "value"])
        self.point_position_key = self._find_key(["target_point", "position"])
        self.last_position = [-4.47, -0.65, -20.0]

    def _find_key(self, required: list[str]) -> str | None:
        for key in self.param_keys:
            low = key.lower()
            if all(part in low for part in required):
                return key
        return None

    def _warmup_model(self) -> None:
        x = torch.zeros(1, self.channels, self.args.image_size, self.args.image_size, device=self.device)
        comp = torch.zeros(1, 3, self.args.image_size, self.args.image_size, device=self.device)
        for _ in range(max(0, self.args.warmup)):
            _ = self.model(x, comp)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _update_light(self, env_rgb: list[float], point_position: list[float], point_rgb: list[float]) -> None:
        if self.env_key is not None:
            self.params[self.env_key] = env_rgb
        if self.point_rgb_key is not None:
            self.params[self.point_rgb_key] = point_rgb
        if self.point_position_key is not None:
            self.params[self.point_position_key] = point_position
        self.params.update()
        self.last_position = list(point_position)

    def _composite(self, render: np.ndarray) -> np.ndarray:
        bg_path = self.scene_dir / "background.png"
        mask_path = self.scene_dir / "geometry_mask.png"
        if not bg_path.exists() or not mask_path.exists():
            return render
        bg = core.read_rgb(bg_path)
        mask = core.read_gray(mask_path)
        if bg.shape[:2] != render.shape[:2]:
            bg_img = Image.fromarray((bg * 255.0).round().astype(np.uint8), "RGB").resize((render.shape[1], render.shape[0]), Image.BICUBIC)
            bg = np.asarray(bg_img, dtype=np.float32) / 255.0
        if mask.shape[:2] != render.shape[:2]:
            mask_img = Image.fromarray((mask[..., 0] * 255.0).round().astype(np.uint8), "L").resize((render.shape[1], render.shape[0]), Image.BICUBIC)
            mask = np.asarray(mask_img, dtype=np.float32)[..., None] / 255.0
        return render * mask + bg * (1.0 - mask)

    def _build_input(self, render: np.ndarray, composite: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        render_t = np_to_chw_tensor(render, self.args.image_size)
        comp_t = np_to_chw_tensor(composite, self.args.image_size)
        tensors = []
        for name in self.input_names:
            if name == "composite":
                tensors.append(comp_t)
            elif name == "render":
                tensors.append(render_t)
            elif name in self.static_tensors:
                tensors.append(self.static_tensors[name])
            else:
                channels = 1 if name in core.GRAY_INPUTS else 3
                tensors.append(torch.zeros(channels, self.args.image_size, self.args.image_size, dtype=torch.float32))
        x = torch.cat(tensors, dim=0)[None, ...].to(self.device, non_blocking=True)
        comp = comp_t[None, ...].to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return x, comp

    @torch.no_grad()
    def render(self, payload: dict) -> dict:
        with self.lock:
            t_total = time.perf_counter()
            env = [float(x) for x in payload.get("env_rgb", [0.2, 0.2, 0.2])]
            pos = [float(x) for x in payload.get("point_light_position", [-4.47, -0.65, -20.0])]
            rgb = [float(x) for x in payload.get("point_light_rgb", [35.0, 32.0, 26.0])]
            spp = int(payload.get("spp", self.args.spp))

            t0 = time.perf_counter()
            self._update_light(env, pos, rgb)
            img = self.mi.render(self.scene, params=self.params, spp=spp, seed=self.seed)
            self.seed += 1
            render = core.linear_to_srgb(np.array(img, dtype=np.float32))
            mitsuba_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            composite = self._composite(render)
            x, comp = self._build_input(render, composite)
            prep_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            pred = self.model(x, comp)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            render_path = self.out_dir / "web_render_in.png"
            composite_path = self.out_dir / "web_composite_in.png"
            pred_path = self.out_dir / "web_efficientvit_prediction.png"
            core.write_rgb(render_path, render)
            core.write_rgb(composite_path, composite)
            core.save_tensor(pred_path, pred[0])
            save_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - t_total) * 1000.0
            return {
                "render_url": f"/output/{render_path.name}",
                "composite_url": f"/output/{composite_path.name}",
                "prediction_url": f"/output/{pred_path.name}",
                "timing": {
                    "mitsuba_ms": mitsuba_ms,
                    "prep_ms": prep_ms,
                    "model_ms": model_ms,
                    "save_ms": save_ms,
                    "total_ms": total_ms,
                    "mitsuba_fps": 1000.0 / mitsuba_ms if mitsuba_ms > 0 else math.inf,
                    "model_fps": 1000.0 / model_ms if model_ms > 0 else math.inf,
                    "total_fps": 1000.0 / total_ms if total_ms > 0 else math.inf,
                },
            }

    def status(self) -> dict:
        return {
            "scene_dir": str(self.scene_dir),
            "out_dir": str(self.out_dir),
            "input_names": self.input_names,
            "channels": self.channels,
            "device": str(self.device),
            "startup_seconds": self.startup_seconds,
            "mitsuba_keys": {
                "env": self.env_key,
                "point_rgb": self.point_rgb_key,
                "point_position": self.point_position_key,
            },
        }


def make_handler(engine: PersistentEngine):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict, status: int = 200) -> None:
            data = json.dumps(obj, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self._send_json(engine.status())
                return
            if parsed.path.startswith("/output/"):
                name = Path(parsed.path).name
                path = engine.out_dir / name
                if not path.exists():
                    self._send_json({"error": f"not found: {name}"}, status=404)
                    return
                self._send_bytes(path.read_bytes(), "image/png")
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/render":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                result = engine.render(payload)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": repr(exc)}, status=500)

        def log_message(self, fmt: str, *args) -> None:
            print("[WEB]", fmt % args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path(r"D:\relighting\Stage3\runs\efficientvit_glassaware_256\checkpoint_last.pt"))
    ap.add_argument("--stage3-root", type=Path, default=Path(r"D:\relighting\Stage3"))
    ap.add_argument("--out-dir", type=Path, default=Path("work/efficientvit_mitsuba_web"))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render-width", type=int, default=256)
    ap.add_argument("--render-height", type=int, default=256)
    ap.add_argument("--glass-mask", type=Path, default=None)
    ap.add_argument("--black-glass-mask", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()

    print("[LOAD] starting persistent engine")
    engine = PersistentEngine(args)
    print("[LOAD] ready in", f"{engine.startup_seconds:.2f}s")
    print("[OPEN]", f"http://{args.host}:{args.port}/")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
