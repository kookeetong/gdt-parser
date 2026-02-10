"""
LLM 接口模块
将提取的区域发送给 LLM 进行识别
"""

import os
import json
import base64
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import aiohttp
import asyncio

from .extractor import ExtractedRegion


@dataclass
class ExtractedDimension:
    """提取的尺寸信息"""
    annotation_id: str
    annotation_type: str  # fai, spc, full_inspection, datum
    nominal: Optional[float] = None
    upper_tolerance: Optional[float] = None
    lower_tolerance: Optional[float] = None
    unit: str = "mm"
    fai_number: Optional[str] = None
    spc_number: Optional[str] = None
    datum_label: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "annotation_id": self.annotation_id,
            "annotation_type": self.annotation_type,
            "nominal": self.nominal,
            "upper_tolerance": self.upper_tolerance,
            "lower_tolerance": self.lower_tolerance,
            "unit": self.unit,
            "fai_number": self.fai_number,
            "spc_number": self.spc_number,
            "datum_label": self.datum_label,
            "raw_text": self.raw_text,
            "confidence": self.confidence
        }


class LLMInterface:
    """LLM 接口"""
    
    PROMPTS = {
        "dimension": """Analyze this engineering drawing annotation and extract the following information:

1. Nominal value (the base dimension number)
2. Upper tolerance (positive deviation)
3. Lower tolerance (negative deviation)  
4. Unit (mm, inch, etc.)

Return as JSON:
{
    "nominal": <float or null>,
    "upper_tolerance": <float or null>,
    "lower_tolerance": <float or null>,
    "unit": "<string>",
    "raw_text": "<original text if readable>",
    "confidence": <0.0-1.0>
}

If you cannot determine a value, use null.""",

        "fai": """This is a First Article Inspection (FAI) annotation from an engineering drawing.

Extract:
1. FAI number/identifier
2. The dimension value and tolerances

Return as JSON:
{
    "fai_number": "<string>",
    "nominal": <float>,
    "upper_tolerance": <float>,
    "lower_tolerance": <float>,
    "unit": "<string>",
    "raw_text": "<original text>",
    "confidence": <0.0-1.0>
}""",

        "spc": """This is a Statistical Process Control (SPC) annotation from an engineering drawing.

Extract:
1. SPC number/identifier
2. The dimension value and tolerances

Return as JSON:
{
    "spc_number": "<string>",
    "nominal": <float>,
    "upper_tolerance": <float>,
    "lower_tolerance": <float>,
    "unit": "<string>",
    "raw_text": "<original text>",
    "confidence": <0.0-1.0>
}""",

        "full_inspection": """This is a 100% inspection annotation from an engineering drawing.

Extract:
1. The dimension value and tolerances

Return as JSON:
{
    "nominal": <float>,
    "upper_tolerance": <float>,
    "lower_tolerance": <float>,
    "unit": "<string>",
    "raw_text": "<original text>",
    "confidence": <0.0-1.0>
}""",

        "datum": """This is a datum (reference) symbol from an engineering drawing.

Extract:
1. Datum label (letter like A, B, C)

Return as JSON:
{
    "datum_label": "<string>",
    "description": "<brief description if visible>",
    "confidence": <0.0-1.0>
}"""
    }
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None
    ):
        """
        初始化 LLM 接口
        
        Args:
            provider: 提供商 (openai, anthropic, local)
            model: 模型名称
            api_key: API 密钥
        """
        self.provider = provider
        self.model = model
        
        if api_key is None:
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        self.api_key = api_key
        
    async def extract_dimension(
        self,
        region: ExtractedRegion,
        annotation_id: str
    ) -> ExtractedDimension:
        """
        从区域提取尺寸信息
        
        Args:
            region: 提取的区域
            annotation_id: 标注 ID
            
        Returns:
            提取的尺寸信息
        """
        # 获取提示词
        prompt_key = region.detection.class_name
        prompt = self.PROMPTS.get(prompt_key, self.PROMPTS["dimension"])
        
        # 调用 LLM
        try:
            result = await self._call_llm(region.base64_image, prompt)
            
            # 解析结果
            data = json.loads(result)
            
            return ExtractedDimension(
                annotation_id=annotation_id,
                annotation_type=region.detection.class_name,
                nominal=data.get("nominal"),
                upper_tolerance=data.get("upper_tolerance"),
                lower_tolerance=data.get("lower_tolerance"),
                unit=data.get("unit", "mm"),
                fai_number=data.get("fai_number"),
                spc_number=data.get("spc_number"),
                datum_label=data.get("datum_label"),
                raw_text=data.get("raw_text"),
                confidence=data.get("confidence", 0.5)
            )
            
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return ExtractedDimension(
                annotation_id=annotation_id,
                annotation_type=region.detection.class_name,
                confidence=0.0
            )
    
    async def batch_extract(
        self,
        regions: List[ExtractedRegion],
        batch_size: int = 5
    ) -> List[ExtractedDimension]:
        """
        批量提取
        
        Args:
            regions: 区域列表
            batch_size: 并发批量大小
            
        Returns:
            提取结果列表
        """
        results = []
        
        for i in range(0, len(regions), batch_size):
            batch = regions[i:i + batch_size]
            tasks = [
                self.extract_dimension(region, f"ann_{i+j:04d}")
                for j, region in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    async def _call_llm(self, image_base64: str, prompt: str) -> str:
        """调用 LLM API"""
        if self.provider == "openai":
            return await self._call_openai(image_base64, prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic(image_base64, prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai(self, image_base64: str, prompt: str) -> str:
        """调用 OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, image_base64: str, prompt: str) -> str:
        """调用 Anthropic API"""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                return data["content"][0]["text"]
