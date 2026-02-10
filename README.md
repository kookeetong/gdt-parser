# GD&T Parser - 工程图纸尺寸标注解析器

> 类似 OmniParser 的 GD&T 工程图纸解析工具，自动识别和提取尺寸标注信息

## 🎯 功能

- **PDF 图纸解析**：支持多页 PDF 工程图纸
- **符号检测**：自动识别 GD&T 相关符号
  - Datum（基准）符号
  - FAI（首件检验）标记
  - SPC（统计过程控制）标记  
  - 100% 测量标记
  - 尺寸标注框
- **区域提取**：给出每个检测元素的 Bounding Box
- **LLM 识别**：将裁剪区域发送给 LLM 提取结构化数据
  - Nominal 值（标称值）
  - Upper Tolerance（上公差）
  - Lower Tolerance（下公差）

## 📐 原理

```
OmniParser 架构                        GD&T Parser 架构
┌────────────────┐                    ┌────────────────┐
│  Screenshot    │                    │  PDF Drawing   │
└───────┬────────┘                    └───────┬────────┘
        │                                     │
        ▼                                     ▼
┌────────────────┐                    ┌────────────────┐
│ icon_detect    │                    │ symbol_detect  │
│ (YOLO)         │                    │ (YOLO/Custom)  │
└───────┬────────┘                    └───────┬────────┘
        │                                     │
        ▼                                     ▼
┌────────────────┐                    ┌────────────────┐
│ icon_caption   │                    │ crop_regions   │
│ (Florence)     │                    │                │
└───────┬────────┘                    └───────┬────────┘
        │                                     │
        ▼                                     ▼
┌────────────────┐                    ┌────────────────┐
│ Structured     │                    │ LLM Extract    │
│ Output         │                    │ (GPT-4V/LLaVA) │
└────────────────┘                    └────────────────┘
```

## 🚀 安装

```bash
# 克隆仓库
git clone https://github.com/kookeetong/gdt-parser.git
cd gdt-parser

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 下载模型权重（可选）
python -m src.download_models
```

## 📖 使用方法

### 命令行

```bash
# 解析单个 PDF
python -m src.gdt_parser parse drawing.pdf --output ./output

# 批量处理
python -m src.gdt_parser batch ./drawings/ --output ./output

# 指定检测类型
python -m src.gdt_parser parse drawing.pdf --types datum fai spc

# 导出 JSON
python -m src.gdt_parser parse drawing.pdf --format json
```

### Python API

```python
from src.gdt_parser import GDTParser

# 初始化解析器
parser = GDTParser()

# 解析 PDF
result = parser.parse("drawing.pdf")

# 访问结果
for page in result.pages:
    print(f"Page {page.number}:")
    for annotation in page.annotations:
        print(f"  Type: {annotation.type}")
        print(f"  BBox: {annotation.bbox}")
        print(f"  Value: {annotation.nominal}")
        print(f"  Tolerance: {annotation.tolerance}")
```

## 📊 输出格式

```json
{
  "file": "drawing.pdf",
  "pages": [
    {
      "number": 1,
      "annotations": [
        {
          "id": "ann_001",
          "type": "fai",
          "bbox": [100, 200, 150, 230],
          "confidence": 0.95,
          "extracted": {
            "fai_number": "FAI-001",
            "nominal": 25.5,
            "upper_tol": 0.1,
            "lower_tol": -0.05,
            "unit": "mm"
          }
        },
        {
          "id": "ann_002",
          "type": "datum",
          "bbox": [200, 300, 250, 340],
          "confidence": 0.92,
          "extracted": {
            "datum_label": "A",
            "description": "Primary datum surface"
          }
        }
      ]
    }
  ]
}
```

## 🔧 配置

创建 `config.yaml`:

```yaml
detection:
  confidence_threshold: 0.7
  nms_threshold: 0.5
  
symbols:
  - datum
  - fai
  - spc
  - full_inspection
  
llm:
  provider: openai  # or local
  model: gpt-4-vision-preview
  api_key: ${OPENAI_API_KEY}
  
output:
  format: json
  include_images: true
  crop_padding: 10
```

## 🧪 测试

```bash
# 运行测试
pytest tests/

# 测试特定功能
pytest tests/test_detector.py -v
```

## 📁 项目结构

```
gdt-parser/
├── src/
│   ├── __init__.py
│   ├── gdt_parser.py      # 主解析器
│   ├── detector.py        # 符号检测
│   ├── extractor.py       # 区域提取
│   ├── llm_interface.py   # LLM 接口
│   ├── pdf_processor.py   # PDF 处理
│   └── utils.py           # 工具函数
├── tests/
│   ├── test_detector.py
│   ├── test_extractor.py
│   └── samples/
├── models/                # 模型权重
├── samples/               # 示例图纸
├── output/                # 输出目录
├── requirements.txt
├── setup.py
├── config.yaml
└── README.md
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 🙏 致谢

- 灵感来源于 [OmniParser](https://github.com/microsoft/OmniParser)
- 使用 YOLO 进行目标检测
- 使用 Florence/BLIP 进行图像描述
