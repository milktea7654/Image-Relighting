from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    root = args.root.resolve()
    handler = partial(Handler, directory=str(root))

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"[OK] serving {root}")
        print(f"[OK] light tuner: http://{args.host}:{args.port}/tools/realtime_light_tuner.html?scene=demo_scene")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
