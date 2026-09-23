from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from huggingface_hub.file_download import get_hf_file_metadata


DEFAULT_REPO = "InternRobotics/InternScenes"
DEFAULT_OUT = Path("compare/internscenes_relighting")
KEY_FILES = [
    "README.md",
    "Scenes_info/scene_name_list_final.json",
    "Layout_info.tar.gz",
    "InternScenes_Gen/Layout_info.tar.gz",
]


def file_metadata(repo_id: str, path: str) -> dict:
    url = hf_hub_url(repo_id, path, repo_type="dataset")
    meta = get_hf_file_metadata(url)
    return {"path": path, "size": meta.size, "etag": meta.etag, "commit_hash": meta.commit_hash}


def try_download(repo_id: str, path: str, out_dir: Path) -> dict:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path,
            local_dir=out_dir / "hf_files",
        )
        return {"path": path, "ok": True, "local_path": str(local_path)}
    except GatedRepoError as exc:
        return {"path": path, "ok": False, "error_type": "GatedRepoError", "error": str(exc).splitlines()[0]}
    except HfHubHTTPError as exc:
        return {"path": path, "ok": False, "error_type": type(exc).__name__, "error": str(exc).splitlines()[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe InternScenes HF access and metadata without downloading full assets.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download-key-files", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    report: dict = {"repo_id": args.repo_id}

    info = api.repo_info(args.repo_id, repo_type="dataset")
    report["repo"] = {
        "sha": info.sha,
        "private": info.private,
        "gated": getattr(info, "gated", None),
    }

    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    report["file_count"] = len(files)
    report["top_level_counts"] = dict(Counter(f.split("/")[0] for f in files).most_common())
    report["second_level_counts"] = dict(Counter("/".join(f.split("/")[:2]) for f in files if "/" in f).most_common(50))
    report["examples"] = files[:120]

    key_file_reports = []
    for path in KEY_FILES:
        try:
            item = file_metadata(args.repo_id, path)
            item["metadata_ok"] = True
        except GatedRepoError as exc:
            item = {
                "path": path,
                "metadata_ok": False,
                "error_type": "GatedRepoError",
                "error": str(exc).splitlines()[0],
            }
        except HfHubHTTPError as exc:
            item = {
                "path": path,
                "metadata_ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc).splitlines()[0],
            }
        key_file_reports.append(item)

    report["key_files"] = key_file_reports

    if args.download_key_files:
        report["downloads"] = [try_download(args.repo_id, path, out_dir) for path in KEY_FILES[:2]]

    report_path = out_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2)[:8000])
    print(f"[done] wrote {report_path}")
    if any(item.get("error_type") == "GatedRepoError" for item in key_file_reports):
        print("[blocked] InternScenes metadata/layout download is gated. Open the HF dataset page and request/accept access, then rerun.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
