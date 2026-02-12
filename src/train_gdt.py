#!/usr/bin/env python3
"""
GD&T YOLO 训练脚本
使用合成数据训练 GD&T 符号检测模型
"""

import os
import random
import yaml
from pathlib import Path
from datetime import datetime


def generate_synthetic_dataset(
    output_dir: str = "datasets/gdt_synthetic",
    num_samples: int = 500,
    img_size: int = 640
):
    """
    生成合成训练数据
    
    Args:
        output_dir: 输出目录
        num_samples: 生成样本数
        img_size: 图像大小
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images" / "train"
    labels_dir = output_path / "labels" / "train"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {num_samples} 个合成样本...")
    
    # GD&T 符号定义 (YOLO format)
    symbols = [
        {"name": "datum", "color": (0, 0, 255), "text": ["A", "B", "C", "D"]},
        {"name": "fai", "color": (0, 255, 0), "text": ["FAI", "F"]},
        {"name": "spc", "color": (255, 0, 0), "text": ["SPC"]},
        {"name": "dimension_box", "color": (0, 255, 255), "text": [""]},
    ]
    
    for i in range(num_samples):
        # 创建空白图像
        img = generate_random_drawing(img_size, symbols)
        
        # 保存图像
        img_path = images_dir / f"img_{i:05d}.png"
        cv2.imwrite(str(img_path), img)
        
        # 生成标注
        label_path = labels_dir / f"img_{i:05d}.txt"
        generate_annotations(img, label_path, symbols)
    
    print(f"✓ 生成完成: {output_dir}")
    
    # 生成 data.yaml
    create_data_yaml(output_path, len(symbols))
    
    return output_path


def generate_random_drawing(size: int, symbols: list) -> np.ndarray:
    """生成随机工程图纸"""
    import numpy as np
    import cv2
    
    # 白色背景 + 网格
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # 绘制工程图框线
    margin = 50
    cv2.rectangle(img, (margin, margin), (size - margin, size - margin), (0, 0, 0), 2)
    
    # 随机绘制一些标题栏
    for _ in range(3):
        x = random.randint(margin, size - margin - 100)
        y = random.randint(margin, size - margin - 50)
        w = random.randint(80, 150)
        h = random.randint(30, 50)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    
    # 随机添加 GD&T 符号
    for symbol in symbols:
        num_symbols = random.randint(2, 8)
        for _ in range(num_symbols):
            x = random.randint(margin + 20, size - margin - 50)
            y = random.randint(margin + 20, size - margin - 50)
            w = random.randint(30, 50)
            h = random.randint(30, 50)
            
            # 绘制符号框
            color = symbol["color"]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # 添加文字
            text = random.choice(symbol["text"])
            if text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
                text_x = x + (w - text_w) // 2
                text_y = y + (h + text_h) // 2
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, 1)
    
    return img


def generate_annotations(img: np.ndarray, label_path: Path, symbols: list):
    """生成 YOLO 格式标注"""
    h, w, _ = img.shape
    
    with open(label_path, 'w') as f:
        for i, symbol in enumerate(symbols):
            # 随机生成一些检测框
            num_boxes = random.randint(2, 8)
            for _ in range(num_boxes):
                x = random.randint(50, w - 100)
                y = random.randint(50, h - 100)
                box_w = random.randint(30, 50)
                box_h = random.randint(30, 50)
                
                # YOLO format: class_id center_x center_y width height (normalized)
                cx = (x + box_w / 2) / w
                cy = (y + box_h / 2) / h
                nw = box_w / w
                nh = box_h / h
                
                f.write(f"{i} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def create_data_yaml(output_path: Path, num_classes: int):
    """创建 data.yaml"""
    yaml_content = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": {
            0: "datum",
            1: "fai",
            2: "spc",
            3: "full_inspection",
            4: "dimension_box"
        }
    }
    
    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f)


def train_yolo(data_yaml: str, epochs: int = 100, batch_size: int = 16):
    """
    训练 YOLO 模型
    
    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批大小
    """
    try:
        from ultralytics import YOLO
        
        print(f"开始训练 YOLO 模型...")
        print(f"  数据: {data_yaml}")
        print(f"  轮数: {epochs}")
        print(f"  批大小: {batch_size}")
        
        # 加载基础模型
        model = YOLO("yolov8n.pt")
        
        # 训练
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=0 if torch.cuda.is_available() else "cpu",
            project="runs/train",
            name="gdt_detector",
            exist_ok=True,
            save_period=10,
            patience=20
        )
        
        print("✓ 训练完成!")
        print(f"  模型保存位置: runs/train/gdt_detector/weights/best.pt")
        
        return results
        
    except ImportError:
        print("✗ 请先安装 ultralytics: pip install ultralytics")


def run_training_workflow():
    """运行完整的训练工作流"""
    print("=" * 60)
    print("GD&T Parser - AI 训练工作流")
    print("=" * 60)
    print()
    
    # 1. 生成合成数据
    print("步骤 1/3: 生成合成训练数据...")
    dataset_dir = generate_synthetic_dataset(
        num_samples=500,
        img_size=640
    )
    
    # 2. 分割训练/验证集
    print("\n步骤 2/3: 分割数据集...")
    split_dataset(dataset_dir)
    
    # 3. 训练模型
    print("\n步骤 3/3: 训练 YOLO 模型...")
    train_yolo(str(dataset_dir / "data.yaml"), epochs=100)
    
    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print()
    print("使用训练好的模型:")
    print("  from src.detector import SymbolDetector")
    print("  detector = SymbolDetector(model_path='runs/train/gdt_detector/weights/best.pt')")


def split_dataset(dataset_path: str, train_ratio: float = 0.8):
    """分割训练/验证集"""
    from pathlib import Path
    import shutil
    
    path = Path(dataset_path)
    train_dir = path / "images" / "train"
    val_dir = path / "images" / "val"
    train_labels = path / "labels" / "train"
    val_labels = path / "labels" / "val"
    
    val_dir.mkdir(exist_ok=True)
    val_labels.mkdir(exist_ok=True)
    
    # 获取所有图像
    images = sorted(train_dir.glob("*.png"))
    
    # 随机选择验证集
    num_val = int(len(images) * (1 - train_ratio))
    val_images = random.sample(images, num_val)
    
    # 移动到验证集
    for img_path in val_images:
        # 移动图像
        shutil.move(str(img_path), str(val_dir / img_path.name))
        # 移动标注
        label_path = train_labels / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels / label_path.name))
    
    print(f"  训练集: {len(list(train_dir.glob('*.png')))} 张图像")
    print(f"  验证集: {len(list(val_dir.glob('*.png')))} 张图像")


if __name__ == "__main__":
    import argparse
    import numpy as np
    import cv2
    import yaml
    import torch
    
    parser = argparse.ArgumentParser(description="GD&T YOLO 训练")
    parser.add_argument("--generate", action="store_true", help="生成合成数据")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--workflow", action="store_true", help="运行完整工作流")
    parser.add_argument("--data", type=str, help="数据集路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    
    args = parser.parse_args()
    
    if args.workflow:
        run_training_workflow()
    elif args.generate:
        generate_synthetic_dataset()
    elif args.train:
        if args.data:
            train_yolo(args.data, args.epochs, args.batch)
        else:
            print("请指定数据集路径: --data <path>")
    else:
        print("用法:")
        print("  python -m src.train_gdt --workflow  # 运行完整工作流")
        print("  python -m src.train_gdt --generate  # 生成合成数据")
        print("  python -m src.train_gdt --train --data <path>  # 训练模型")
