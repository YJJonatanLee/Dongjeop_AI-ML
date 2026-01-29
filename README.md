# Vision Models Multi-Label Classification

다양한 Vision 모델(SigLIP, DINOv2, ConvNeXt 등)을 사용한 multi-label image classification 파인튜닝 프로젝트.

**지원 모델**: SigLIP, SigLIP2, DINOv2, ConvNeXt, ConvNeXt V2, SAM-2, EVA-02
**지원 Head**: Linear, ML-Decoder, CSRA, Q2L

## 태스크

실내 이미지에서 다음 5가지 클래스의 존재 여부를 예측:

- `has_step`: 계단/턱 존재 여부
- `has_movable_chair`: 이동 가능한 의자 존재 여부
- `has_high_chair`: 높은 의자 존재 여부
- `has_fixed_chair`: 고정된 의자 존재 여부
- `has_floor_chair`: 바닥 좌석 존재 여부

## 프로젝트 구조

```
Dongjeop_Classification/
├── src/
│   ├── training/
│   │   ├── train.py          # 학습 메인 스크립트
│   │   ├── trainer.py        # Trainer 클래스
│   │   └── losses.py         # Custom loss functions (ASL, BCE)
│   ├── models/
│   │   ├── model_loader.py   # 모델 로더 (모든 backbone + head 지원)
│   │   └── heads/            # Classification Heads
│   │       ├── ml_decoder.py # ML-Decoder
│   │       ├── csra.py       # CSRA (Class-Specific Residual Attention)
│   │       └── q2l.py        # Q2L (Query2Label)
│   └── dataloader/
│       └── dataset.py        # Multi-label dataset
├── infer/
│   └── classifier.py         # 추론 클래스
├── models/
│   ├── configs/              # 모델별 설정 파일 (YAML)
│   └── checkpoints/          # 학습된 모델 저장 (backbone/head별 분류)
│       └── {model-name}/
│           ├── linear/
│           ├── mldecoder/
│           ├── csra/
│           └── q2l/
├── outputs/logs/             # 학습 로그 (backbone/head별 분류)
│   └── {model-name}/
│       ├── linear/
│       ├── mldecoder/
│       ├── csra/
│       └── q2l/
├── run_training.sh           # 학습 실행 스크립트
└── run_infer.sh              # 추론 실행 스크립트
```

## Quick Start

### 1. 환경 설정

```bash
# uv 사용 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.10

# 또는 pip 사용
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# 도움말 보기
./run_training.sh --help

# DINOv2 + ML Decoder (추천!)
./run_training.sh --model-type dinov2-mldecoder

# SigLIP + CSRA
./run_training.sh --model-type siglip-csra

# SigLIP2 + Q2L
./run_training.sh --model-type siglip2-q2l

# ConvNeXt + ML Decoder
./run_training.sh --model-type convnext-mldecoder

# Linear head (빠른 실험)
./run_training.sh --model-type dinov2
./run_training.sh --model-type siglip

# 워커 수 지정
./run_training.sh --model-type dinov2-mldecoder --num-workers 8
```

### 3. 추론 실행

```bash
# 기본 모델로 추론
./run_infer.sh --image test.jpg

# 파인튜닝된 모델로 추론
./run_infer.sh --image test.jpg \
  --checkpoint models/checkpoints/siglip-base-patch16-224/best_model_*.pth

# 임계값 조정
./run_infer.sh --image test.jpg \
  --checkpoint models/checkpoints/.../best_model_*.pth \
  --threshold 0.6
```

## 지원 모델 조합

### Backbone 모델 (7개)

| 모델 | Model ID | 아키텍처 |
|------|----------|---------|
| SigLIP | `google/siglip-base-patch16-224` | Vision Transformer |
| SigLIP2 | `google/siglip2-base-patch16-224` | Improved ViT |
| DINOv2 | `facebook/dinov2-base` | Self-supervised ViT |
| ConvNeXt | `facebook/convnext-base-224-22k` | Modern CNN |
| ConvNeXt V2 | `facebook/convnextv2-base-1k-224` | CNN + FCMAE |
| SAM-2 | `facebook/sam2-hiera-base-plus` | Hiera (Hierarchical ViT) |
| EVA-02 | `timm/eva02_base_patch14_224.mim_in22k` | ViT + MIM |

### Classification Head (4개)

| Head | 특징 | 파라미터 |
|------|------|---------|
| Linear | 간단한 선형 계층 | ~4K |
| ML-Decoder | Transformer decoder, Query embeddings | ~7.9M |
| CSRA | Class-specific attention + Residual | ~3M |
| Q2L | Transformer decoder, Label correlation | ~16M |

### 사용 가능한 조합 (28가지)

```bash
# SigLIP 계열
siglip, siglip-mldecoder, siglip-csra, siglip-q2l
siglip2, siglip2-mldecoder, siglip2-csra, siglip2-q2l

# DINOv2 계열
dinov2, dinov2-mldecoder, dinov2-csra, dinov2-q2l

# ConvNeXt 계열
convnext, convnext-mldecoder, convnext-csra, convnext-q2l

# ConvNeXt V2 계열
convnextv2, convnextv2-mldecoder, convnextv2-csra, convnextv2-q2l

# SAM-2 계열
sam2, sam2-mldecoder, sam2-csra, sam2-q2l

# EVA-02 계열
eva02, eva02-mldecoder, eva02-csra, eva02-q2l
```

## 데이터셋

### 데이터 형식

```json
{
  "has_step": false,
  "has_movable_chair": true,
  "has_high_chair": false,
  "has_fixed_chair": false,
  "has_floor_chair": false,
  "image_id": "20250915_130118_87196638",
  "image": "20250915_130118_87196638.jpg"
}
```

### 데이터 경로 설정

`models/configs/*.yaml` 파일에서 데이터 경로 설정:

```yaml
data:
  train_ann_file: "train_images/train_annotation.json"
  train_img_dir: "train_images/images"
  val_ann_file: "train_images/val_annotation.json"
  val_img_dir: "train_images/images"
```

## 주요 설정

### Config 파일 예시

```yaml
model:
  pretrained_model_id: google/siglip-base-patch16-224
  num_labels: 5

# Classification Head 설정 (하나만 enabled: true로 설정)
ml_decoder:
  enabled: false
  num_queries: 5
  num_layers: 1
  num_heads: 8

csra:
  enabled: false
  num_heads: 4
  lam: 0.1
  temperatures: [1.0, 2.0, 4.0, 6.0]

q2l:
  enabled: false
  num_layers: 2
  num_heads: 8

training:
  epochs: 60
  batch_size: 32
  learning_rate: 5.0e-5
  vision_encoder_lr: 1.0e-6  # Differential LR
  classifier_lr: 3.5e-4
  early_stopping_patience: 8
  best_metric: "mAP_macro"

  loss:
    type: "asl"  # 'asl' or 'bce'
    params:
      gamma_neg: 4
      gamma_pos: 1
      clip: 0.05
```

## 평가 메트릭

- **mAP (Mean Average Precision)**: 주요 메트릭 (macro/micro)
- **F1 Score** (Micro/Macro): Multi-label F1 점수
- **Subset Accuracy**: 모든 레이블이 정확히 일치하는 비율
- **Hamming Accuracy**: 레이블별 정확도 평균

## 주요 기능

### 1. Multi-Head 지원
- **ML-Decoder**: Transformer 기반 multi-label classification
- **CSRA**: Class-specific attention + residual connection
- **Q2L**: Query2Label - label correlation 모델링

### 2. Differential Learning Rate
Vision encoder와 classifier에 다른 learning rate 적용
```yaml
vision_encoder_lr: 1.0e-6  # 낮은 LR (pretrained 보존)
classifier_lr: 3.5e-4      # 높은 LR (새로운 head 학습)
```

### 3. Loss Functions
- **ASL (Asymmetric Loss)**: Hard negative mining (권장)
- **BCE Loss**: 기본 binary cross-entropy

### 4. Mixed Precision Training
자동 mixed precision (AMP) 지원

### 5. Early Stopping & Best Model Selection
다양한 메트릭 기반 early stopping

## Python API

### 추론

```python
from infer.classifier import SigLIPClassifier

# 분류기 초기화
classifier = SigLIPClassifier(
    model_id="facebook/dinov2-base",
    checkpoint_path="models/checkpoints/.../best_model.pth",
    threshold=0.5
)

# 단일 이미지 분류
result = classifier.classify("test.jpg")
print(result)
# {
#   'has_step': False,
#   'has_step_prob': 0.123,
#   'has_movable_chair': True,
#   'has_movable_chair_prob': 0.876,
#   ...
# }

# 배치 분류
results = classifier.batch_classify(["img1.jpg", "img2.jpg"])
```

## 참고 자료

### Models
- [SigLIP Paper](https://arxiv.org/abs/2303.15343) - Sigmoid Loss for Language Image Pre-Training
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) - Self-supervised Vision Transformers
- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545) - A ConvNet for the 2020s
- [SAM-2 Paper](https://arxiv.org/abs/2408.00714) - Segment Anything in Images and Videos
- [EVA-02 Paper](https://arxiv.org/abs/2303.11331) - EVA-02: A Visual Representation for Neon Genesis

### Classification Heads
- [ML-Decoder Paper](https://arxiv.org/abs/2111.12933) - Scalable Multi-Label Classification
- [CSRA Paper (ICCV 2021)](https://arxiv.org/abs/2108.02456) - Residual Attention for Multi-Label Recognition
- [Q2L Paper](https://arxiv.org/abs/2107.10834) - Query2Label: A Simple Transformer Way

### Techniques
- [ASL Paper](https://arxiv.org/abs/2009.14119) - Asymmetric Loss for Multi-Label Classification

## License

MIT License
