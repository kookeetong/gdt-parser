# Issue #1 改进方案

## 问题描述
使用 `yolov8n.pt` 基础模型时，检测结果为空 (`annotations: []`)

## 原因分析
- 基础 YOLO 模型未针对 GD&T 符号训练
- 无法识别图纸中的基准符号、FAI 标记等

## 解决方案

### 方案 1: 下载预训练模型 (推荐)

```bash
# 下载 OmniParser 预训练权重
python -m src.download_models
```

这将下载:
- OmniParser 模型 (用于图标检测)
- GroundingDINO (用于开放词汇检测)

### 方案 2: 使用增强版检测器

```python
from src.detector_v2 import EnhancedDetector

# 模式选择: auto / yolo / grounding / pattern / llm
detector = EnhancedDetector(
    mode="pattern",  # 基于传统CV，无需模型
    confidence_threshold=0.3
)

# 检测图像
detections = detector.detect(image)
```

### 方案 3: 训练自己的模型

#### 3.1 生成合成训练数据

```bash
# 生成 500 张带标注的合成图纸
python -m src.train_gdt --generate
```

#### 3.2 训练 YOLO 模型

```bash
# 训练 100 轮
python -m src.train_gdt --train --data datasets/gdt_synthetic/data.yaml --epochs 100
```

#### 3.3 使用训练好的模型

```python
from src.detector import SymbolDetector

detector = SymbolDetector(
    model_path="runs/train/gdt_detector/weights/best.pt"
)
```

### 方案 4: 使用 LLM 辅助检测

如果安装了 LLM 接口，可以使用 GPT-4V 等视觉模型辅助检测：

```python
from src.llm_interface import LLMDetector

detector = LLMDetector(provider="openai", model="gpt-4-vision")
detections = detector.detect(image)
```

## 快速开始

```bash
# 1. 克隆并安装
git clone https://github.com/kookeetong/gdt-parser.git
cd gdt-parser
pip install -r requirements.txt

# 2. 尝试模式检测 (无需模型)
python -c "
from src.detector_v2 import EnhancedDetector
detector = EnhancedDetector(mode='pattern')
image = cv2.imread('samples/test.pdf')
detections = detector.detect(image)
print(f'检测到 {len(detections)} 个符号')
"

# 3. 或训练自己的模型
python -m src.train_gdt --workflow
```

## 输出示例

使用增强版检测器后，输出应包含检测到的符号:

```json
{
  "file": "data/test.pdf",
  "pages": [
    {
      "number": 1,
      "annotations": [
        {
          "id": "ann_001",
          "type": "datum",
          "bbox": [100, 200, 130, 230],
          "confidence": 0.65,
          "extracted": null
        },
        {
          "id": "ann_002",
          "type": "dimension_box",
          "bbox": [150, 200, 250, 230],
          "confidence": 0.58,
          "extracted": null
        }
      ]
    }
  ]
}
```

## 注意事项

1. **基础 YOLO 模型**: `yolov8n.pt` 未针对 GD&T 训练，结果可能为空
2. **模式检测**: 准确率较低，适合快速测试
3. **LLM 检测**: 需要 API key，成本较高
4. **自定义训练**: 推荐方案，需要准备标注数据

## 下一步

1. 准备真实标注数据 (使用 labelImg 等工具)
2. 使用 `python -m src.train_gdt --generate` 生成数据模板
3. 训练模型并调整参数
4. 评估准确率并迭代优化
