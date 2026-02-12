#!/usr/bin/env python3
"""
GD&T 数据生成器 - 生成合成训练数据
无需 GPU，直接运行生成 YOLO 格式数据集
"""

import os
import random
import json
from pathlib import Path
from datetime import datetime


def generate_synthetic_dataset(
    output_dir: str = "datasets/gdt_synthetic",
    num_samples: int = 500,
    img_size: int = 640
):
    """
    生成合成训练数据
    
    输出格式:
    - images/train/*.png
    - labels/train/*.txt (YOLO format)
    """
    import numpy as np
    import cv2
    
    output_path = Path(output_dir)
    images_dir = output_path / "images" / "train"
    labels_dir = output_path / "labels" / "train"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {num_samples} 个合成样本...")
    
    # GD&T 符号定义 (class_id -> name)
    CLASSES = {
        0: "datum",           # 基准符号 (方形框内字母)
        1: "fai",             # FAI 标记
        2: "spc",             # SPC 标记
        3: "full_inspection",  # 100% 检验
        4: "dimension_box",   # 尺寸标注框
    }
    
    for i in range(num_samples):
        # 创建空白图像
        img = generate_random_drawing(img_size, CLASSES)
        
        # 保存图像
        img_path = images_dir / f"img_{i:05d}.png"
        cv2.imwrite(str(img_path), img)
        
        # 生成标注 (YOLO format)
        label_path = labels_dir / f"img_{i:05d}.txt"
        generate_annotations(img, label_path, CLASSES)
        
        if (i + 1) % 50 == 0:
            print(f"  已完成 {i + 1}/{num_samples}")
    
    print(f"✓ 生成完成!")
    print(f"  图像: {images_dir}")
    print(f"  标注: {labels_dir}")
    
    # 生成 data.yaml
    create_data_yaml(output_path, CLASSES)
    
    return output_path


def generate_random_drawing(size: int, classes: dict) -> "np.ndarray":
    """生成随机工程图纸"""
    import numpy as np
    import cv2
    
    # 白色背景
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # 绘制图纸边框
    margin = 50
    cv2.rectangle(img, (margin, margin), (size - margin, size - margin), (0, 0, 0), 2)
    
    # 绘制标题栏
    for _ in range(3):
        x = random.randint(margin, size - margin - 150)
        y = random.randint(margin, size - margin - 60)
        w = random.randint(100, 200)
        h = random.randint(40, 60)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
        
        # 标题栏内添加文字
        for _ in range(random.randint(1, 4)):
            text_x = x + random.randint(5, w - 30)
            text_y = y + random.randint(15, h - 10)
            cv2.putText(img, str(random.randint(0, 999)), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # 绘制尺寸标注线
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(margin + 20, size - margin - 50)
        y1 = random.randint(margin + 100, size - margin - 50)
        length = random.randint(50, 150)
        angle = random.choice([0, 45, 90])
        
        if angle == 0:
            x2 = x1 + length
            y2 = y1
        elif angle == 45:
            x2 = x1 + int(length * 0.7)
            y2 = y1 - int(length * 0.7)
        else:
            x2 = x1
            y2 = y1 - length
        
        cv2.arrowedLine(img, (x1, y1), (x2, y2), (50, 50, 50), 1, tipLength=0.1)
    
    # 随机添加 GD&T 符号
    for class_id, class_name in classes.items():
        num_symbols = random.randint(3, 10)
        for _ in range(num_symbols):
            x = random.randint(margin + 20, size - margin - 80)
            y = random.randint(margin + 20, size - margin - 80)
            w = random.randint(30, 60)
            h = random.randint(30, 60)
            
            # 绘制符号框
            color = {
                0: (0, 0, 255),      # datum: 红色
                1: (0, 255, 0),      # fai: 绿色
                2: (255, 0, 0),      # spc: 蓝色
                3: (255, 255, 0),    # full_inspection: 青色
                4: (0, 255, 255),    # dimension_box: 黄色
            }[class_id]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # 添加文字标签
            text_map = {
                0: random.choice(["A", "B", "C", "D", "E"]),
                1: random.choice(["FAI", "F", "F-AI"]),
                2: random.choice(["SPC"]),
                3: random.choice(["100%", "全检"]),
                4: "",
            }
            text = text_map[class_id]
            if text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                text_x = x + (w - tw) // 2
                text_y = y + (h + th) // 2
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, 1)
    
    return img


def generate_annotations(img: "np.ndarray", label_path: Path, classes: dict):
    """生成 YOLO 格式标注"""
    h, w, _ = img.shape
    
    with open(label_path, 'w') as f:
        # 遍历所有类
        for class_id in classes.keys():
            # 查找该类的区域 (简化: 随机生成)
            num_boxes = random.randint(3, 10)
            boxes = set()  # 去重
            
            for _ in range(num_boxes):
                x = random.randint(50, w - 100)
                y = random.randint(50, h - 100)
                box_w = random.randint(30, 60)
                box_h = random.randint(30, 60)
                
                # YOLO format: class_id cx cy w h (归一化)
                cx = (x + box_w / 2) / w
                cy = (y + box_h / 2) / h
                nw = box_w / w
                nh = box_h / h
                
                # 避免重复
                key = (round(cx, 2), round(cy, 2))
                if key not in boxes:
                    boxes.add(key)
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def create_data_yaml(output_path: Path, classes: dict):
    """创建 data.yaml"""
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_path.absolute()}
train: images/train
val: images/val

nc: {len(classes)}
names:
"""
    
    for class_id, class_name in classes.items():
        yaml_content += f"  {class_id}: {class_name}\n"
    
    with open(output_path / "data.yaml", 'w') as f:
        f.write(yaml_content)


def generate_sample_images():
    """生成示例图像用于手动标注"""
    output_dir = "datasets/sample_for_labeling"
    output_path = Path(output_dir)
    
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    
    print(f"生成 10 张示例图像用于标注...")
    
    # 只生成少量样本
    for i in range(10):
        img = generate_random_drawing(640, {0: "datum", 4: "dimension_box"})
        cv2.imwrite(str(output_path / "images" / f"sample_{i:02d}.png"), img)
    
    print(f"✓ 示例图像已保存到: {output_dir}")
    print(f"  请使用 labelImg 等工具进行标注，然后运行训练")


def run_full_workflow():
    """运行完整数据生成工作流"""
    print("=" * 60)
    print("GD&T Parser - 训练数据生成")
    print("=" * 60)
    print()
    
    # 1. 生成合成数据
    print("步骤 1/2: 生成合成训练数据...")
    dataset_dir = generate_synthetic_dataset(num_samples=500)
    
    # 2. 生成示例图像用于手动标注
    print()
    print("步骤 2/2: 生成示例图像用于手动标注...")
    generate_sample_images()
    
    print()
    print("=" * 60)
    print("数据生成完成!")
    print("=" * 60)
    print()
    print("下一步:")
    print("1. 查看 datasets/gdt_synthetic 中的合成数据")
    print("2. 使用 labelImg 标注真实数据 (可选):")
    print("   pip install labelImg")
    print("   labelImg datasets/sample_for_labeling/images")
    print("3. 训练 YOLO 模型:")
    print("   cd datasets/gdt_synthetic")
    print("   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100")
    print()
    print("或在 Google Colab 上训练:")
    print("1. 上传 datasets/gdt_synthetic 到 Google Drive")
    print("2. 打开 https://colab.research.google.com/")
    print("3. 运行 YOLO 训练")


if __name__ == "__main__":
    import argparse
    import numpy as np
    import cv2
    
    parser = argparse.ArgumentParser(description="GD&T 训练数据生成")
    parser.add_argument("--samples", type=int, default=500, help="生成样本数")
    parser.add_argument("--size", type=int, default=640, help="图像大小")
    parser.add_argument("--output", type=str, default="datasets/gdt_synthetic", help="输出目录")
    parser.add_argument("--workflow", action="store_true", help="运行完整工作流")
    
    args = parser.parse_args()
    
    if args.workflow:
        run_full_workflow()
    else:
        generate_synthetic_dataset(args.output, args.samples, args.size)
        print()
        print("数据已生成!")
        print(f"  位置: {args.output}")
        print()
        print("训练 YOLO:")
        print(f"  cd {args.output}")
        print("  yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100")
