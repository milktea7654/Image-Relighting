#!/usr/bin/env python3
"""下載缺失的檔案: initial_light.json, optimize_config.json, scene_mitsuba.json"""

import json
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://140.113.121.155:8777/"
JSONL_FILE = Path("test.jsonl")
DOWNLOAD_DIR = Path("downloaded_data")

# 需要下載的額外檔案
EXTRA_FILES = [
    "light/initial_light.json",
    "optimize_config.json", 
    "scene_mitsuba.json"
]

def download_file(url: str, output_path: Path) -> bool:
    """下載單個文件"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"  ✓ {output_path.name}")
        return True
    except urllib.error.HTTPError as e:
        if e.code != 404:  # 不要對404報錯
            print(f"  ✗ HTTP {e.code}: {url}")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def main():
    if not JSONL_FILE.exists():
        print(f"ERROR: {JSONL_FILE} not found")
        return
    
    # 從 test.jsonl 提取所有 scene_id
    scene_ids = set()
    with open(JSONL_FILE) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                scene_id = record.get('scene_id')
                if scene_id:
                    scene_ids.add(scene_id)
    
    print(f"Found {len(scene_ids)} scenes")
    print(f"Downloading {len(EXTRA_FILES)} extra files per scene...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for scene_id in sorted(scene_ids):
        for extra_file in EXTRA_FILES:
            url = f"{BASE_URL}{scene_id}/{extra_file}"
            output_path = DOWNLOAD_DIR / scene_id / extra_file
            
            if download_file(url, output_path):
                success_count += 1
            else:
                failed_count += 1
    
    print(f"\nSummary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {success_count + failed_count}")

if __name__ == '__main__':
    main()
