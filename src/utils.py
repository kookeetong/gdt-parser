"""
工具函数
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def resize_image(
    image: np.ndarray,
    max_size: int = 1920
) -> np.ndarray:
    """调整图像大小"""
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    return image


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 10
) -> np.ndarray:
    """带边距裁剪"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2].copy()


def draw_grid(
    image: np.ndarray,
    grid_size: int = 100,
    color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """绘制网格"""
    result = image.copy()
    h, w = image.shape[:2]
    
    for x in range(0, w, grid_size):
        cv2.line(result, (x, 0), (x, h), color, 1)
    
    for y in range(0, h, grid_size):
        cv2.line(result, (0, y), (w, y), color, 1)
    
    return result


def merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """合并重叠的边界框"""
    if not boxes:
        return []
    
    boxes = list(boxes)
    merged = []
    
    while boxes:
        current = boxes.pop(0)
        
        changed = True
        while changed:
            changed = False
            new_boxes = []
            
            for box in boxes:
                if iou(current, box) > threshold:
                    current = (
                        min(current[0], box[0]),
                        min(current[1], box[1]),
                        max(current[2], box[2]),
                        max(current[3], box[3])
                    )
                    changed = True
                else:
                    new_boxes.append(box)
            
            boxes = new_boxes
        
        merged.append(current)
    
    return merged


def iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def create_summary_image(
    original: np.ndarray,
    detections: List,
    max_width: int = 1200
) -> np.ndarray:
    """创建带标注摘要的图像"""
    # 调整原始图像大小
    scale = min(max_width / original.shape[1], 1.0)
    if scale < 1.0:
        resized = cv2.resize(original, None, fx=scale, fy=scale)
    else:
        resized = original.copy()
    
    # 绘制检测结果
    for det in detections:
        x1, y1, x2, y2 = [int(v * scale) for v in det.bbox]
        cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(resized, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return resized
