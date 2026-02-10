"""
PDF 处理模块
将 PDF 图纸转换为图像
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PDFPage:
    """PDF 页面"""
    number: int
    image: np.ndarray
    width: int
    height: int
    dpi: int


class PDFProcessor:
    """PDF 处理器"""
    
    def __init__(
        self,
        dpi: int = 300,
        image_format: str = "png"
    ):
        """
        初始化 PDF 处理器
        
        Args:
            dpi: 渲染 DPI
            image_format: 输出图像格式
        """
        self.dpi = dpi
        self.image_format = image_format
    
    def load_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[PDFPage]:
        """
        加载 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            pages: 要加载的页码列表（None 表示所有页）
            
        Returns:
            页面列表
        """
        doc = fitz.open(pdf_path)
        result = []
        
        total_pages = len(doc)
        
        if pages is None:
            pages = list(range(total_pages))
        
        for page_num in pages:
            if page_num >= total_pages:
                continue
                
            page = doc[page_num]
            
            # 渲染页面为图像
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为 numpy 数组
            img_data = pix.tobytes(self.image_format)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result.append(PDFPage(
                number=page_num + 1,
                image=image,
                width=pix.width,
                height=pix.height,
                dpi=self.dpi
            ))
        
        doc.close()
        return result
    
    def save_page_images(
        self,
        pages: List[PDFPage],
        output_dir: str,
        prefix: str = "page"
    ) -> List[str]:
        """
        保存页面图像
        
        Args:
            pages: 页面列表
            output_dir: 输出目录
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for page in pages:
            filename = f"{prefix}_{page.number:04d}.{self.image_format}"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), page.image)
            saved_paths.append(str(filepath))
        
        return saved_paths
    
    def get_page_info(self, pdf_path: str) -> dict:
        """
        获取 PDF 信息
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            PDF 信息字典
        """
        doc = fitz.open(pdf_path)
        
        info = {
            "path": pdf_path,
            "pages": len(doc),
            "metadata": doc.metadata,
            "page_sizes": []
        }
        
        for page in doc:
            rect = page.rect
            info["page_sizes"].append({
                "width": rect.width,
                "height": rect.height
            })
        
        doc.close()
        return info
    
    def extract_text_regions(
        self,
        pdf_path: str,
        page_num: int = 0
    ) -> List[dict]:
        """
        提取文本区域（用于辅助检测）
        
        Args:
            pdf_path: PDF 文件路径
            page_num: 页码
            
        Returns:
            文本区域列表
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        blocks = page.get_text("dict")["blocks"]
        regions = []
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        regions.append({
                            "text": span["text"],
                            "bbox": list(bbox),
                            "font": span["font"],
                            "size": span["size"]
                        })
        
        doc.close()
        return regions
