#!/usr/bin/env python3
"""驗證並下載所有缺失的檔案"""

import json
import urllib.request
import urllib.error
from pathlib import Path
from html.parser import HTMLParser
import re

BASE_URL = "http://140.113.121.155:8777/"
JSONL_FILE = Path("test.jsonl")
DOWNLOAD_DIR = Path("downloaded_data")

class LinkExtractor(HTMLParser):
    """從 HTML 中提取檔案和文件夾連結"""
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value and not value.startswith('?'):
                    self.links.append(value)

def get_server_files(scene_id: str, subfolder: str = "") -> set:
    """取得伺服器上該資料夾內的所有檔案和文件夾"""
    if subfolder:
        url = f"{BASE_URL}{scene_id}/{subfolder}/"
    else:
        url = f"{BASE_URL}{scene_id}/"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            html = response.read().decode('utf-8')
            parser = LinkExtractor()
            parser.feed(html)
            # 過濾掉父目錄連結和空值
            links = {link for link in parser.links if link and link != '../'}
            return links
    except Exception as e:
        print(f"ERROR fetching {url}: {e}")
        return set()

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
        return True
    except Exception as e:
        return False

def main():
    if not JSONL_FILE.exists():
        print(f"ERROR: {JSONL_FILE} not found")
        return
    
    # 從 test.jsonl 提取所有 scene_id
    scene_ids = []
    with open(JSONL_FILE) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                scene_id = record.get('scene_id')
                if scene_id:
                    scene_ids.append(scene_id)
    
    print(f"檢查 {len(scene_ids)} 個 scenes...")
    print()
    
    total_missing = 0
    total_downloaded = 0
    scene_with_issues = []
    
    for i, scene_id in enumerate(scene_ids, 1):
        # 取得伺服器上的檔案
        server_files = get_server_files(scene_id)
        server_files_light = get_server_files(scene_id, "light")
        
        if not server_files and not server_files_light:
            continue
        
        local_dir = DOWNLOAD_DIR / scene_id
        local_light_dir = local_dir / "light"
        
        missing_files = []
        
        # 檢查主目錄檔案
        for file in server_files:
            if not file.endswith('/'):  # 只檢查檔案，不是文件夾
                local_path = local_dir / file
                if not local_path.exists():
                    missing_files.append(('', file))
        
        # 檢查 light 子目錄檔案
        for file in server_files_light:
            if not file.endswith('/'):
                local_path = local_light_dir / file
                if not local_path.exists():
                    missing_files.append(('light', file))
        
        if missing_files:
            scene_with_issues.append((scene_id, missing_files))
            total_missing += len(missing_files)
            
            # 下載缺失的檔案
            for subfolder, filename in missing_files:
                if subfolder:
                    url = f"{BASE_URL}{scene_id}/{subfolder}/{filename}"
                    output_path = local_dir / subfolder / filename
                else:
                    url = f"{BASE_URL}{scene_id}/{filename}"
                    output_path = local_dir / filename
                
                if download_file(url, output_path):
                    total_downloaded += 1
        
        if i % 100 == 0:
            print(f"  已檢查 {i}/{len(scene_ids)} scenes...")
    
    print()
    print("=" * 60)
    print(f"找到 {total_missing} 個缺失檔案")
    print(f"成功下載 {total_downloaded} 個檔案")
    print(f"{len(scene_with_issues)} 個 scenes 有缺失檔案")
    
    if scene_with_issues and len(scene_with_issues) <= 20:
        print()
        print("缺失檔案詳情:")
        for scene_id, missing in scene_with_issues[:10]:
            print(f"  {scene_id}:")
            for subfolder, filename in missing[:3]:
                path = f"{subfolder}/{filename}" if subfolder else filename
                print(f"    - {path}")
            if len(missing) > 3:
                print(f"    ... 還有 {len(missing) - 3} 個檔案")

if __name__ == '__main__':
    main()
