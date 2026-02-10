"""
GD&T Parser - 工程图纸尺寸标注解析器
类似 OmniParser 的 GD&T 图纸解析工具
"""

__version__ = "0.1.0"
__author__ = "OpenClaw"

from .gdt_parser import GDTParser
from .detector import SymbolDetector
from .extractor import RegionExtractor
from .llm_interface import LLMInterface

__all__ = [
    "GDTParser",
    "SymbolDetector", 
    "RegionExtractor",
    "LLMInterface"
]
