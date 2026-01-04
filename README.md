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
│   │   └── trainer.py        # Trainer 클래스
│   ├── models/
│   │   └── model_loader.py   # SigLIP 모델 로더 (LoRA 지원)
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
# SigLIP 학습
./run_training.sh --model-type siglip

# SigLIP2 학습
./run_training.sh --model-type siglip2

# 커스텀 설정으로 학습
./run_training.sh --config models/configs/my_config.yaml --num-workers 8
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
  epochs: 30
  batch_size: 32
  learning_rate: 5.0e-5
  warmup_epochs: 3
  early_stopping_patience: 5
  freeze_vision_encoder: false

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
```

## 📈 평가 메트릭

- **Subset Accuracy**: 모든 레이블이 정확히 일치하는 비율
- **Hamming Accuracy**: 레이블별 정확도 평균
- **F1 Score** (Micro/Macro): Multi-label F1 점수
- **Precision/Recall** (Micro/Macro)

## 💡 주요 기능

### 1. LoRA Fine-tuning
PEFT 라이브러리를 사용한 효율적인 파인튜닝

### 2. Mixed Precision Training
자동 mixed precision (AMP) 지원으로 메모리 절약

### 3. Early Stopping
Validation loss 기반 조기 종료

### 4. Data Augmentation
Albumentations 기반 다양한 증강 기법

### 5. Multi-label Classification
Sigmoid + BCE Loss를 사용한 multi-label 예측

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
./run_training.sh --model-type siglip --num-workers 4
./run_training.sh --model-type siglip2 --num-workers 4

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

## 🆚 SigLIP vs SigLIP2

| 특징 | SigLIP | SigLIP2 |
|------|--------|---------|
| **성능** | 우수 | 더 우수 |
| **메모리** | 효율적 | 효율적 |
| **학습 속도** | 빠름 | 빠름 |
| **추천** | 일반적인 경우 | 최고 성능 필요 시 |

## 📝 TODO

- [ ] SigLIP2 테스트
- [ ] Test 데이터셋 평가 스크립트 추가
- [ ] Wandb 로깅 추가
- [ ] Gradio 데모 추가

## 📚 참고 자료

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [SigLIP2 Hugging Face](https://huggingface.co/blog/siglip2)
- [Multi-Label Classification Tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb)

## 📄 License

MIT License
