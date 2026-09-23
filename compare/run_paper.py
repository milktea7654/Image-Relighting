from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent


def load_methods() -> dict[str, dict[str, Any]]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return data["tool"]["compare"]["methods"]


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def list_methods(methods: dict[str, dict[str, Any]]) -> int:
    print("Available compare methods:")
    for name in sorted(methods):
        method = methods[name]
        print(f"- {name}: {method['title']}")
        print(f"  extra: {method['extra']}")
        print(f"  cwd: {method['cwd']}")
        print(f"  paper: {method['paper']}")
    return 0


def run_method(method_name: str, extra_args: list[str]) -> int:
    methods = load_methods()
    if method_name not in methods:
        print(f"Unknown method: {method_name}", file=sys.stderr)
        print("Run 'python run_paper.py list' to see available methods.", file=sys.stderr)
        return 2

    method = methods[method_name]
    cwd = ROOT / method["cwd"]
    entry = cwd / method["entry"]
    if not cwd.is_dir():
        print(f"Missing directory: {cwd}", file=sys.stderr)
        return 2
    if not entry.is_file():
        print(f"Missing entry script: {entry}", file=sys.stderr)
        return 2

    python = method.get("python")
    if python:
        python_path = ROOT / python
        if not python_path.is_file():
            print(f"Missing method Python: {python_path}", file=sys.stderr)
            return 2
        executable = str(python_path)
    else:
        executable = sys.executable

    command = [executable, str(Path(method["entry"])), *method.get("default_args", []), *extra_args]
    env = os.environ.copy()
    if python:
        env["VIRTUAL_ENV"] = str((ROOT / python).parents[1])
        env["PATH"] = str((ROOT / python).parent) + os.pathsep + env.get("PATH", "")
    blender_dir = ROOT / "tools" / "blender-4.1.1-windows-x64"
    if blender_dir.is_dir():
        env["PATH"] = str(blender_dir) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[compare] method: {method_name}")
    print(f"[compare] extra: {method['extra']}")
    if python:
        print(f"[compare] python: {python_path}")
    print(f"[compare] cwd: {cwd}")
    print(f"[compare] paper: {method['paper']}")
    print(f"[compare] code: {method['code']}")
    print(f"[compare] command: {format_command(command)}")
    if method_name == "scriblit":
        print("[compare] ScribbleLight expects pretrained weights under compare/scriblit/scribblelight_controlnet/.")
    if method_name.startswith("luminet"):
        print("[compare] LumiNet weights are gated on Hugging Face. Accept access at https://huggingface.co/xyxingx/LumiNet before running.")

    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode


def main() -> int:
    methods = load_methods()
    parser = argparse.ArgumentParser(
        description="Run cloned comparison papers from the compare workspace.",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="list",
        help="Method name to run, or 'list' to show available methods.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the upstream script. Use '--' before them if needed.",
    )
    args = parser.parse_args()

    if args.method == "list":
        return list_methods(methods)

    extra_args = args.extra_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    return run_method(args.method, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
