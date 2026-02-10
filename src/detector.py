"""
符号检测模块
使用 YOLO 模型检测 GD&T 相关符号
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import torch


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int
    
    def to_dict(self) -> Dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_name": self.class_name,
            "class_id": self.class_id
        }


class SymbolDetector:
    """GD&T 符号检测器"""
    
    # 符号类别定义
    SYMBOL_CLASSES = {
        0: "datum",          # 基准符号
        1: "fai",            # FAI 标记
        2: "spc",            # SPC 标记
        3: "full_inspection", # 100% 检验
        4: "dimension_box",   # 尺寸标注框
        5: "tolerance",       # 公差标注
        6: "surface_finish",  # 表面粗糙度
        7: "geometric_tol",   # 形位公差
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        初始化检测器
        
        Args:
            model_path: YOLO 模型路径（如果为 None，使用预训练模型）
            confidence_threshold: 置信度阈值
            nms_threshold: NMS 阈值
            device: 运行设备 (auto/cpu/cuda)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 加载模型
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # 使用基础 YOLO 模型（需要后续微调）
            self.model = YOLO("yolov8n.pt")
            print("Warning: Using base YOLO model. For GD&T detection, please train or provide a fine-tuned model.")
        
        self.model.to(self.device)
        
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Detection]:
        """
        检测图像中的 GD&T 符号
        
        Args:
            image: 输入图像 (BGR 格式)
            classes: 要检测的类别列表（None 表示所有类别）
            
        Returns:
            检测结果列表
        """
        # YOLO 推理
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # 获取边界框
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 获取类别和置信度
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 获取类别名称
                class_name = self.SYMBOL_CLASSES.get(class_id, "unknown")
                
                # 过滤类别
                if classes and class_name not in classes:
                    continue
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name=class_name,
                    class_id=class_id
                )
                detections.append(detection)
        
        return detections
    
    def detect_from_file(
        self,
        image_path: str,
        classes: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        从文件检测
        
        Args:
            image_path: 图像文件路径
            classes: 要检测的类别列表
            
        Returns:
            (图像, 检测结果)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        detections = self.detect(image, classes)
        return image, detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果
            color_map: 类别颜色映射
            
        Returns:
            绘制后的图像
        """
        if color_map is None:
            color_map = {
                "datum": (255, 0, 0),      # 蓝色
                "fai": (0, 255, 0),         # 绿色
                "spc": (0, 0, 255),         # 红色
                "full_inspection": (255, 255, 0),  # 青色
                "dimension_box": (255, 0, 255),    # 紫色
                "tolerance": (0, 255, 255),        # 黄色
                "surface_finish": (128, 128, 0),
                "geometric_tol": (128, 0, 128),
            }
        
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 标签背景
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # 标签文字
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result_image
    
    def fine_tune(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640
    ):
        """
        微调模型
        
        Args:
            data_yaml: 数据集配置文件
            epochs: 训练轮数
            batch_size: 批大小
            img_size: 图像大小
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device
        )


class PatternBasedDetector:
    """
    基于模式的检测器（无需训练的备用方案）
    使用模板匹配和传统 CV 方法检测 GD&T 符号
    """
    
    def __init__(self):
        self.templates = {}
        
    def load_template(self, name: str, template_path: str):
        """加载模板"""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            self.templates[name] = template
            
    def detect_datum(self, image: np.ndarray) -> List[Detection]:
        """检测基准符号（方形框内字母）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 查找方形轮廓
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # 基准符号通常是正方形
            if 0.8 < aspect_ratio < 1.2 and 20 < w < 50:
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.7,
                    class_name="datum",
                    class_id=0
                ))
        
        return detections
    
    def detect_fai_pattern(self, image: np.ndarray) -> List[Detection]:
        """检测 FAI 标记"""
        # FAI 通常以 "FAI" 或 "F" 开头
        # 使用 OCR 或模式匹配
        
        # 简化版本：查找特定形状
        detections = []
        
        # TODO: 实现更精确的检测逻辑
        
        return detections
