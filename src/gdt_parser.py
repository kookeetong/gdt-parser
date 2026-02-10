"""
GD&T Parser - 主解析器
整合所有模块，提供完整的解析流程
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .detector import SymbolDetector, Detection
from .extractor import RegionExtractor, ExtractedRegion
from .llm_interface import LLMInterface, ExtractedDimension
from .pdf_processor import PDFProcessor, PDFPage


@dataclass
class Annotation:
    """完整的标注信息"""
    id: str
    page_number: int
    type: str
    bbox: List[int]
    confidence: float
    extracted: Optional[Dict] = None
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class ParseResult:
    """解析结果"""
    file: str
    pages: int
    annotations: List[Annotation]
    processing_time: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "file": self.file,
            "pages": self.pages,
            "annotations": [a.to_dict() for a in self.annotations],
            "processing_time": self.processing_time,
            "timestamp": self.timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class GDTParser:
    """GD&T 图纸解析器"""
    
    def __init__(
        self,
        detector_model: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4-vision-preview",
        llm_api_key: Optional[str] = None,
        confidence_threshold: float = 0.5,
        dpi: int = 300,
        output_dir: str = "./output"
    ):
        """
        初始化解析器
        
        Args:
            detector_model: 检测模型路径
            llm_provider: LLM 提供商
            llm_model: LLM 模型名称
            llm_api_key: LLM API 密钥
            confidence_threshold: 检测置信度阈值
            dpi: PDF 渲染 DPI
            output_dir: 输出目录
        """
        self.detector = SymbolDetector(
            model_path=detector_model,
            confidence_threshold=confidence_threshold
        )
        
        self.extractor = RegionExtractor()
        
        self.llm = LLMInterface(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key
        )
        
        self.pdf_processor = PDFProcessor(dpi=dpi)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        symbols: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> ParseResult:
        """
        解析 PDF 图纸
        
        Args:
            pdf_path: PDF 文件路径
            pages: 要解析的页码（None 表示所有页）
            symbols: 要检测的符号类型（None 表示所有）
            use_llm: 是否使用 LLM 提取详细信息
            
        Returns:
            解析结果
        """
        import time
        start_time = time.time()
        
        # 1. 加载 PDF
        pdf_pages = self.pdf_processor.load_pdf(pdf_path, pages)
        
        # 2. 保存页面图像
        page_images_dir = self.output_dir / "pages"
        page_paths = self.pdf_processor.save_page_images(
            pdf_pages, str(page_images_dir)
        )
        
        all_annotations = []
        
        for page in pdf_pages:
            # 3. 检测符号
            detections = self.detector.detect(page.image, symbols)
            
            # 4. 提取区域
            regions = self.extractor.extract(page.image, detections)
            
            # 5. 保存裁剪区域
            crops_dir = self.output_dir / "crops" / f"page_{page.number:04d}"
            crop_paths = self.extractor.save_regions(regions, str(crops_dir))
            
            # 6. LLM 提取（如果启用）
            if use_llm and self.llm.api_key:
                try:
                    dimensions = asyncio.run(
                        self.llm.batch_extract(regions)
                    )
                except Exception as e:
                    print(f"LLM extraction failed: {e}")
                    dimensions = []
            else:
                dimensions = []
            
            # 7. 组装结果
            for i, (region, crop_path) in enumerate(zip(regions, crop_paths)):
                extracted = None
                if i < len(dimensions):
                    extracted = dimensions[i].to_dict()
                
                annotation = Annotation(
                    id=f"ann_p{page.number:02d}_{i:04d}",
                    page_number=page.number,
                    type=region.detection.class_name,
                    bbox=list(region.detection.bbox),
                    confidence=region.detection.confidence,
                    extracted=extracted,
                    image_path=crop_path
                )
                all_annotations.append(annotation)
        
        processing_time = time.time() - start_time
        
        return ParseResult(
            file=pdf_path,
            pages=len(pdf_pages),
            annotations=all_annotations,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def parse_image(
        self,
        image_path: str,
        symbols: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> Dict:
        """
        解析单个图像
        
        Args:
            image_path: 图像文件路径
            symbols: 要检测的符号类型
            use_llm: 是否使用 LLM
            
        Returns:
            解析结果
        """
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测
        detections = self.detector.detect(image, symbols)
        
        # 提取
        regions = self.extractor.extract(image, detections)
        
        # LLM
        if use_llm and self.llm.api_key:
            try:
                dimensions = asyncio.run(self.llm.batch_extract(regions))
            except:
                dimensions = []
        else:
            dimensions = []
        
        # 组装
        annotations = []
        for i, (region, dim) in enumerate(zip(regions, dimensions)):
            annotations.append({
                "id": f"ann_{i:04d}",
                "type": region.detection.class_name,
                "bbox": region.detection.bbox,
                "confidence": region.detection.confidence,
                "extracted": dim.to_dict() if dim else None
            })
        
        # 绘制可视化
        vis_image = self.detector.draw_detections(image, detections)
        vis_path = str(self.output_dir / f"visualization_{Path(image_path).stem}.png")
        cv2.imwrite(vis_path, vis_image)
        
        return {
            "file": image_path,
            "annotations": annotations,
            "visualization": vis_path
        }
    
    def batch_parse(
        self,
        input_dir: str,
        pattern: str = "*.pdf",
        **kwargs
    ) -> List[ParseResult]:
        """
        批量解析
        
        Args:
            input_dir: 输入目录
            pattern: 文件模式
            **kwargs: 传递给 parse() 的参数
            
        Returns:
            解析结果列表
        """
        input_path = Path(input_dir)
        results = []
        
        for file_path in input_path.glob(pattern):
            print(f"Processing: {file_path}")
            try:
                result = self.parse(str(file_path), **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
        
        return results
    
    def export_results(
        self,
        results: List[ParseResult],
        output_file: str,
        format: str = "json"
    ):
        """
        导出结果
        
        Args:
            results: 结果列表
            output_file: 输出文件路径
            format: 输出格式 (json, csv, excel)
        """
        output_path = Path(output_file)
        
        if format == "json":
            data = [r.to_dict() for r in results]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
                
        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "file", "page", "id", "type", 
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "confidence", "nominal", "upper_tol", "lower_tol"
                ])
                for result in results:
                    for ann in result.annotations:
                        ext = ann.extracted or {}
                        writer.writerow([
                            result.file, ann.page_number, ann.id, ann.type,
                            *ann.bbox,
                            ann.confidence,
                            ext.get("nominal", ""),
                            ext.get("upper_tolerance", ""),
                            ext.get("lower_tolerance", "")
                        ])
        
        print(f"Results exported to: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GD&T Parser")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--pages", "-p", type=int, nargs="+", help="Page numbers to process")
    parser.add_argument("--symbols", "-s", nargs="+", help="Symbol types to detect")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM extraction")
    parser.add_argument("--format", "-f", choices=["json", "csv"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    gdt_parser = GDTParser(output_dir=args.output)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = gdt_parser.parse(
            str(input_path),
            pages=args.pages,
            symbols=args.symbols,
            use_llm=not args.no_llm
        )
        results = [result]
    else:
        results = gdt_parser.batch_parse(
            str(input_path),
            pages=args.pages,
            symbols=args.symbols,
            use_llm=not args.no_llm
        )
    
    # 导出结果
    output_file = Path(args.output) / f"results.{args.format}"
    gdt_parser.export_results(results, str(output_file), args.format)
    
    print(f"\nProcessed {len(results)} file(s)")
    print(f"Total annotations: {sum(len(r.annotations) for r in results)}")


if __name__ == "__main__":
    main()
