"""
区域提取模块
从检测结果中裁剪区域并准备发送给 LLM
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import base64
from io import BytesIO

from .detector import Detection


@dataclass
class ExtractedRegion:
    """提取的区域"""
    detection: Detection
    cropped_image: np.ndarray
    base64_image: Optional[str] = None
    context_image: Optional[np.ndarray] = None  # 带上下文的图像
    
    def to_dict(self) -> Dict:
        return {
            "bbox": self.detection.to_dict(),
            "image_size": self.cropped_image.shape[:2],
            "base64": self.base64_image
        }


class RegionExtractor:
    """区域提取器"""
    
    def __init__(
        self,
        padding: int = 10,
        context_padding: int = 50,
        min_size: int = 20
    ):
        """
        初始化提取器
        
        Args:
            padding: 裁剪区域的额外边距
            context_padding: 上下文区域的边距
            min_size: 最小区域大小
        """
        self.padding = padding
        self.context_padding = context_padding
        self.min_size = min_size
    
    def extract(
        self,
        image: np.ndarray,
        detections: List[Detection],
        include_context: bool = True
    ) -> List[ExtractedRegion]:
        """
        从图像中提取检测区域
        
        Args:
            image: 源图像
            detections: 检测结果
            include_context: 是否包含上下文图像
            
        Returns:
            提取的区域列表
        """
        h, w = image.shape[:2]
        regions = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 添加边距
            x1_p = max(0, x1 - self.padding)
            y1_p = max(0, y1 - self.padding)
            x2_p = min(w, x2 + self.padding)
            y2_p = min(h, y2 + self.padding)
            
            # 检查最小尺寸
            if (x2_p - x1_p) < self.min_size or (y2_p - y1_p) < self.min_size:
                continue
            
            # 裁剪区域
            cropped = image[y1_p:y2_p, x1_p:x2_p].copy()
            
            # 生成 base64
            base64_img = self._image_to_base64(cropped)
            
            # 提取上下文
            context = None
            if include_context:
                cx1 = max(0, x1 - self.context_padding)
                cy1 = max(0, y1 - self.context_padding)
                cx2 = min(w, x2 + self.context_padding)
                cy2 = min(h, y2 + self.context_padding)
                context = image[cy1:cy2, cx1:cx2].copy()
            
            region = ExtractedRegion(
                detection=det,
                cropped_image=cropped,
                base64_image=base64_img,
                context_image=context
            )
            regions.append(region)
        
        return regions
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将图像转换为 base64"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def extract_connected(
        self,
        image: np.ndarray,
        detections: List[Detection],
        group_distance: int = 100
    ) -> List[ExtractedRegion]:
        """
        提取相关联的区域（如尺寸标注及其对应的公差）
        
        Args:
            image: 源图像
            detections: 检测结果
            group_distance: 分组距离阈值
            
        Returns:
            提取的关联区域
        """
        if not detections:
            return []
        
        # 分组检测
        groups = self._group_detections(detections, group_distance)
        
        regions = []
        h, w = image.shape[:2]
        
        for group in groups:
            # 计算组的边界框
            all_x1 = [d.bbox[0] for d in group]
            all_y1 = [d.bbox[1] for d in group]
            all_x2 = [d.bbox[2] for d in group]
            all_y2 = [d.bbox[3] for d in group]
            
            x1 = max(0, min(all_x1) - self.padding)
            y1 = max(0, min(all_y1) - self.padding)
            x2 = min(w, max(all_x2) + self.padding)
            y2 = min(h, max(all_y2) + self.padding)
            
            cropped = image[y1:y2, x1:x2].copy()
            base64_img = self._image_to_base64(cropped)
            
            # 使用主要检测类型
            primary_det = max(group, key=lambda d: d.bbox[2] - d.bbox[0])
            
            region = ExtractedRegion(
                detection=primary_det,
                cropped_image=cropped,
                base64_image=base64_img
            )
            regions.append(region)
        
        return regions
    
    def _group_detections(
        self,
        detections: List[Detection],
        max_distance: int
    ) -> List[List[Detection]]:
        """将相近的检测分组"""
        if not detections:
            return []
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            cx1 = (det1.bbox[0] + det1.bbox[2]) / 2
            cy1 = (det1.bbox[1] + det1.bbox[3]) / 2
            
            for j, det2 in enumerate(detections):
                if j in used:
                    continue
                
                cx2 = (det2.bbox[0] + det2.bbox[2]) / 2
                cy2 = (det2.bbox[1] + det2.bbox[3]) / 2
                
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                
                if dist < max_distance:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def save_regions(
        self,
        regions: List[ExtractedRegion],
        output_dir: str,
        prefix: str = "region"
    ) -> List[str]:
        """
        保存提取的区域到文件
        
        Args:
            regions: 区域列表
            output_dir: 输出目录
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, region in enumerate(regions):
            filename = f"{prefix}_{region.detection.class_name}_{i:04d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), region.cropped_image)
            saved_paths.append(str(filepath))
        
        return saved_paths
    
    def create_visualization(
        self,
        image: np.ndarray,
        regions: List[ExtractedRegion]
    ) -> np.ndarray:
        """创建可视化图像，显示所有提取的区域"""
        vis_image = image.copy()
        
        for region in regions:
            x1, y1, x2, y2 = region.detection.bbox
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制序号
            cv2.putText(
                vis_image,
                f"{region.detection.class_name}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return vis_image
