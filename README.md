# SigLIP Multi-Label Classification

SigLIP/SigLIP2 모델을 사용한 multi-label image classification 파인튜닝 프로젝트.

## 📋 태스크

실내 이미지에서 다음 5가지 클래스의 존재 여부를 예측:

- `has_step`: 계단/턱 존재 여부
- `has_movable_chair`: 이동 가능한 의자 존재 여부
- `has_high_chair`: 높은 의자 존재 여부
- `has_fixed_chair`: 고정된 의자 존재 여부
- `has_floor_chair`: 바닥 좌석 존재 여부

## 🏗️ 프로젝트 구조

```
SigLIP_Classification/
├── src/
│   ├── training/
│   │   ├── train.py          # 학습 메인 스크립트
│   │   ├── trainer.py        # Trainer 클래스
│   │   └── losses.py         # Custom loss functions (ASL, Focal)
│   ├── models/
│   │   ├── model_loader.py   # SigLIP 모델 로더
│   │   └── ml_decoder.py     # ML-Decoder implementation
│   └── dataloader/
│       └── dataset.py        # Multi-label dataset
├── infer/
│   └── classifier.py         # 추론 클래스
├── models/
│   ├── configs/
│   │   ├── siglip_config.yaml   # SigLIP 설정
│   │   └── siglip2_config.yaml  # SigLIP2 설정
│   └── checkpoints/          # 학습된 모델 저장
├── outputs/logs/             # 학습 로그
├── run_training.sh          # 학습 실행 스크립트
└── run_infer.sh            # 추론 실행 스크립트
```

## 🚀 Quick Start

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
# SigLIP 학습 (linear head)
./run_training.sh --model-type siglip

# SigLIP + ML Decoder 학습 (권장!)
./run_training.sh --model-type siglip-mldecoder

# SigLIP2 학습 (linear head)
./run_training.sh --model-type siglip2

# SigLIP2 + ML Decoder 학습 (최고 성능!)
./run_training.sh --model-type siglip2-mldecoder

# 커스텀 설정으로 학습
./run_training.sh --config models/configs/my_config.yaml --num-workers 8
```

### 3. 추론 실행

```bash
# 기본 모델로 추론
./run_infer.sh --image test.jpg

# 파인튜닝된 linear head 모델로 추론
./run_infer.sh --image test.jpg \
  --checkpoint models/checkpoints/siglip-base-patch16-224/best_model_*.pth

# ML Decoder 모델로 추론 (자동 감지)
./run_infer.sh --image test.jpg \
  --checkpoint models/checkpoints/siglip-base-patch16-224-mldecoder/best_model_*.pth

# 임계값 조정
./run_infer.sh --image test.jpg \
  --checkpoint models/checkpoints/siglip-base-patch16-224/best_model_*.pth \
  --threshold 0.6
```

## 📊 데이터셋

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

## ⚙️ 주요 설정

### SigLIP 설정 예시

```yaml
model:
  pretrained_model_id: google/siglip-base-patch16-224
  num_labels: 5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 5.0e-5
  warmup_epochs: 10
  early_stopping_patience: 15
  freeze_vision_encoder: false  # Full fine-tuning
  best_metric: "mAP_macro"

  loss:
    type: "asl"  # ASL, Focal, BCE
    params:
      gamma_neg: 4
      gamma_pos: 1
      clip: 0.05
```

## 📈 평가 메트릭

- **Subset Accuracy**: 모든 레이블이 정확히 일치하는 비율
- **Hamming Accuracy**: 레이블별 정확도 평균
- **F1 Score** (Micro/Macro): Multi-label F1 점수
- **Precision/Recall** (Micro/Macro)

## 💡 주요 기능

### 1. ML-Decoder Support with Feature Maps (NEW!)
Transformer 기반 multi-label classification head로 클래스 간 관계 모델링
- **Spatial Feature Maps 사용**: 196개 spatial tokens (14x14 grid)을 직접 활용
- **Query embeddings + Cross-attention**: 각 클래스가 이미지의 특정 공간 영역에 attend
- 기존 linear head 대비 성능 향상
- 7.9M 파라미터 추가
- Pooled output이 아닌 전체 feature map을 사용하여 공간 정보 보존

### 2. Full Model Fine-tuning
전체 모델 end-to-end 학습으로 최고 성능 달성
- Vision encoder + Classification head 동시 학습
- `freeze_vision_encoder: false` 설정

### 3. Advanced Loss Functions
클래스 불균형 문제를 해결하는 다양한 loss 지원
- **ASL (Asymmetric Loss)**: Hard negative mining (권장)
- **Focal Loss**: 어려운 샘플에 집중
- **BCE Loss**: 기본 binary cross-entropy

### 4. Mixed Precision Training
자동 mixed precision (AMP) 지원으로 메모리 절약 및 학습 속도 향상

### 5. Early Stopping & Best Model Selection
다양한 메트릭 기반 early stopping
- mAP (macro/micro), F1 score, Subset accuracy 등 지원
- Validation 기반 최적 모델 자동 저장

### 6. Data Augmentation
Albumentations 기반 다양한 증강 기법
- HorizontalFlip, ColorJitter, Affine transformation 등

## 🔧 Python API 사용

### 학습

```python
from src.models.model_loader import load_model
from src.training.trainer import SigLIPTrainer

# 모델 로드
model = load_model(config_path="models/configs/siglip_config.yaml")

# 학습 (자세한 내용은 src/training/train.py 참고)
```
### Quick start
```bash
# SigLIP + ML Decoder (권장)
./run_training.sh --model-type siglip-mldecoder --num-workers 4

# SigLIP2 + ML Decoder (최고 성능)
./run_training.sh --model-type siglip2-mldecoder --num-workers 4

# Linear head 모델들
./run_training.sh --model-type siglip --num-workers 4
./run_training.sh --model-type siglip2 --num-workers 4
```

### 추론

```python
from infer.classifier import SigLIPClassifier

# 분류기 초기화
classifier = SigLIPClassifier(
    model_id="google/siglip-base-patch16-224",
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

## 🆚 모델 비교

### Classification Head 비교

| 특징 | Linear Head | ML Decoder |
|------|------------|-----------|
| **파라미터** | ~4K | 7.9M |
| **클래스 관계 모델링** | ✗ | ✓ (Cross-attention) |
| **학습 속도** | 빠름 | 중간 |
| **성능** | 기준 | 향상 (특히 mAP) |
| **추천** | 빠른 실험 | 최종 성능 |

### Backbone 비교

| 특징 | SigLIP | SigLIP2 |
|------|--------|---------|
| **성능** | 우수 | 더 우수 |
| **메모리** | 효율적 | 효율적 |
| **학습 속도** | 빠름 | 빠름 |
| **추천** | 일반적인 경우 | 최고 성능 필요 시 |

**권장 조합**:
- **빠른 학습**: SigLIP + ML Decoder
- **최고 성능**: SigLIP2 + ML Decoder

## 📝 TODO

- [x] ML Decoder 통합
- [x] ML Decoder에서 spatial feature maps 사용 (성능 개선)
- [ ] SigLIP2 + ML Decoder 조합 테스트
- [ ] Test 데이터셋 평가 스크립트 추가
- [ ] Wandb 로깅 추가
- [ ] Gradio 데모 추가

## 📚 참고 자료

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [SigLIP2 Hugging Face](https://huggingface.co/blog/siglip2)
- [ML-Decoder Paper](https://arxiv.org/abs/2111.12933)
- [ML-Decoder Official Implementation](https://github.com/Alibaba-MIIL/ML_Decoder)
- [Multi-Label Classification Tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb)

## 📄 License

MIT License
