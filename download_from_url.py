#!/usr/bin/env python3
"""下載test.jsonl中指定的文件"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

# 配置
BASE_URL = "http://140.113.121.155:8777/"
JSONL_FILE = Path("test.jsonl")
DOWNLOAD_DIR = Path("downloaded_data")

def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """下載單個文件"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(output_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (chunk_size * 100) == 0:
                        print(f"  Downloaded: {downloaded / 1024 / 1024:.1f} MB")
        
        print(f"  ✓ Success: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def main():
    if not JSONL_FILE.exists():
        print(f"ERROR: {JSONL_FILE} not found")
        sys.exit(1)
    
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    # 提取所有需要下載的文件路徑
    files_to_download = set()
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                scene_id = record.get('scene_id')
                # 添加所有相對路徑字段，將 output/ 前綴替換為 scene_id/
                for key in ['reference', 'render_optimized', 'composite_optimized', 
                           'residual', 'manifest', 'optimization_log', 'optimized_light', 
                           'optimized_point_lights']:
                    if key in record and record[key]:
                        file_path = record[key]
                        # 將 output/scene_XXXXX/ 改為 scene_XXXXX/
                        if 'output/' in file_path and scene_id:
                            file_path = file_path.replace('output/', '')
                        files_to_download.add(file_path)
            except json.JSONDecodeError as e:
                print(f"ERROR parsing line {line_no}: {e}")
    
    print(f"Found {len(files_to_download)} unique file paths to download")
    
    # 下載文件
    success_count = 0
    failed_count = 0
    
    for file_path in sorted(files_to_download):
        url = urljoin(BASE_URL, file_path)
        output_path = DOWNLOAD_DIR / file_path
        
        print(f"\nDownloading: {file_path}")
        print(f"From: {url}")
        
        if download_file(url, output_path):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n\nSummary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {success_count + failed_count}")
    print(f"Downloaded to: {DOWNLOAD_DIR.resolve()}")

if __name__ == '__main__':
    main()
