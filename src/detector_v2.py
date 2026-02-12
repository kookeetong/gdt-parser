#!/usr/bin/env python3
"""
GD&T 符号检测器 - 增强版
支持多种检测模式：YOLO / GroundingDINO / 传统CV
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch

# 尝试导入可选依赖
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import groundingdino
    from groundingdino.util import box_ops
    GROUNDING_AVAILABLE = True
except ImportError:
    GROUNDING_AVAILABLE = False


@dataclass
class Detection:
    """检测结果"""
    bbox: Tuple[int, int, int, int]
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


class EnhancedDetector:
    """增强版 GD&T 检测器"""
    
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
        mode: str = "auto",  # auto / yolo / grounding / pattern / llm
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,  # 降低阈值
        device: str = "auto"
    ):
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 根据模式初始化检测器
        self._init_detector(model_path)
    
    def _init_detector(self, model_path: Optional[str]):
        """初始化检测器"""
        if self.mode == "auto":
            # 自动选择最佳可用模式
            if YOLO_AVAILABLE:
                self.mode = "yolo"
            elif GROUNDING_AVAILABLE:
                self.mode = "grounding"
            else:
                self.mode = "pattern"
        
        if self.mode == "yolo":
            self._init_yolo(model_path)
        elif self.mode == "grounding":
            self._init_grounding(model_path)
        elif self.mode == "pattern":
            self._init_pattern()
        elif self.mode == "llm":
            self._init_llm()
    
    def _init_yolo(self, model_path: Optional[str]):
        """初始化 YOLO"""
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"✓ 加载自定义模型: {model_path}")
        elif YOLO_AVAILABLE:
            # 尝试加载 OmniParser 模型
            omni_model = Path(__file__).parent.parent / "models" / "omni" / "icon_detect" / "weights" / "best.pt"
            if omni_model.exists():
                self.model = YOLO(str(omni_model))
                print(f"✓ 加载 OmniParser 模型")
            else:
                self.model = YOLO("yolov8n.pt")
                print("⚠ 使用基础 YOLO 模型 (未针对 GD&T 训练)")
        else:
            raise ImportError("请安装 ultralytics: pip install ultralytics")
        
        self.model.to(self.device)
    
    def _init_grounding(self, model_path: Optional[str]):
        """初始化 GroundingDINO"""
        if not GROUNDING_AVAILABLE:
            raise ImportError("请安装 GroundingDINO: pip install groundingdino")
        
        # 加载模型
        self.grounding_model = groundingdino.GroundingDINO.build_from_filename(
            str(Path(__file__).parent.parent / "models" / "grounding" / "groundingdino_swint_ogc.pth")
        )
        print("✓ 加载 GroundingDINO 模型")
    
    def _init_pattern(self):
        """初始化传统 CV 检测器"""
        print("✓ 使用基于模式的检测器 (无需模型)")
    
    def _init_llm(self):
        """初始化 LLM 检测器"""
        print("✓ 使用 LLM 辅助检测")
    
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Detection]:
        """检测 GD&T 符号"""
        if self.mode == "yolo":
            return self._detect_yolo(image, classes)
        elif self.mode == "grounding":
            return self._detect_grounding(image, classes)
        elif self.mode == "pattern":
            return self._detect_pattern(image, classes)
        elif self.mode == "llm":
            return self._detect_llm(image, classes)
        return []
    
    def _detect_yolo(self, image, classes) -> List[Detection]:
        """YOLO 检测"""
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append(Detection(
                    bbox=tuple(map(int, xyxy)),
                    confidence=float(box.conf[0]),
                    class_name=self.SYMBOL_CLASSES.get(int(box.cls[0]), "unknown"),
                    class_id=int(box.cls[0])
                ))
        
        return detections
    
    def _detect_grounding(self, image, classes) -> List[Detection]:
        """GroundingDINO 检测"""
        prompt = "datum symbol, FAI mark, SPC symbol, dimension box, tolerance value"
        detections = []
        
        # 预处理图像
        h, w, _ = image.shape
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 检测
        boxes = self.grounding_model.detect(image_tensor, prompt, "")
        
        # 转换格式
        for i in range(len(boxes)):
            xyxy = boxes[i].cpu().numpy()
            detections.append(Detection(
                bbox=tuple(map(int, xyxy)),
                confidence=0.5,
                class_name="unknown",
                class_id=0
            ))
        
        return detections
    
    def _detect_pattern(self, image, classes) -> List[Detection]:
        """基于模式的检测 (备用方案)"""
        detections = []
        
        # 检测基准符号 (方形框)
        if not classes or "datum" in classes:
            detections.extend(self._detect_datum_pattern(image))
        
        # 检测尺寸标注框
        if not classes or "dimension_box" in classes:
            detections.extend(self._detect_dimension_pattern(image))
        
        return detections
    
    def _detect_datum_pattern(self, image) -> List[Detection]:
        """检测基准符号模式"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            
            # 基准符号通常是 1:1 正方形
            if 0.7 < aspect < 1.3 and 15 < w < 60:
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.6,
                    class_name="datum",
                    class_id=0
                ))
        
        return detections
    
    def _detect_dimension_pattern(self, image) -> List[Detection]:
        """检测尺寸标注框"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 水平 + 垂直线检测
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
        
        # 合并
        grid = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
        
        # 阈值
        _, thresh = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找框
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 30 < w < 200 and 10 < h < 50:
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5,
                    class_name="dimension_box",
                    class_id=4
                ))
        
        return detections
    
    def _detect_llm(self, image, classes) -> List[Detection]:
        """LLM 辅助检测 (使用视觉 LLM)"""
        # TODO: 实现 LLM 检测
        # 可以使用 GPT-4V 或 LLaVA
        return []


def create_detector(mode: str = "auto", **kwargs) -> EnhancedDetector:
    """工厂方法"""
    return EnhancedDetector(mode=mode, **kwargs)
