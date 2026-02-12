#!/usr/bin/env python3
"""
模型下载脚本
下载预训练的 GD&T 检测模型
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# 模型下载地址
MODELS = {
    "omni_parser": {
        "url": "https://github.com/microsoft/OmniParser/releases/download/v1.0/omni_parser_weights.zip",
        "description": "OmniParser weights for icon detection",
        "local_dir": "models/omni"
    },
    "gdt_yolo": {
        "url": None,  # 待提供
        "description": "Fine-tuned YOLO model for GD&T symbols",
        "local_dir": "models/gdt"
    },
    " grounding_dino": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0.2/groundingdino_swint_ogc.pth",
        "description": "GroundingDINO for detection",
        "local_dir": "models/grounding"
    }
}

def download_file(url, dest_path, desc="Downloading"):
    """下载文件"""
    if os.path.exists(dest_path):
        print(f"✓ {desc} already exists")
        return True
    
    try:
        print(f"↓ {desc}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ {desc} saved to {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {desc}: {e}")
        return False

def download_models():
    """下载所有模型"""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("GD&T Parser - 模型下载")
    print("=" * 50)
    print()
    
    # 下载 OmniParser
    omni_dir = models_dir / "omni"
    omni_dir.mkdir(exist_ok=True)
    omni_zip = omni_dir / "weights.zip"
    
    if download_file(
        MODELS["omni_parser"]["url"],
        str(omni_zip),
        "OmniParser weights"
    ):
        # 解压
        print("解压 OmniParser weights...")
        with zipfile.ZipFile(omni_zip, 'r') as zip_ref:
            zip_ref.extractall(omini_dir)
        print("✓ 解压完成")
    
    # 下载 GroundingDINO
    grounding_dir = models_dir / "grounding"
    grounding_dir.mkdir(exist_ok=True)
    grounding_path = grounding_dir / "groundingdino_swint_ogc.pth"
    
    download_file(
        MODELS["grounding_dino"]["url"],
        str(grounding_path),
        "GroundingDINO weights"
    )
    
    print()
    print("=" * 50)
    print("模型下载完成!")
    print("=" * 50)
    print()
    print("提示: 如果需要 GD&T 专用模型，请:")
    print("1. 准备标注数据 (YOLO format)")
    print("2. 运行: python -m src.train_gdt")
    print("3. 或者联系我们获取预训练权重")

if __name__ == "__main__":
    download_models()
