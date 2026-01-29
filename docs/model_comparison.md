# Model Comparison: 7 Backbone Models

실내 객체 인식 multi-label classification을 위한 모델 선택 가이드

## Quick Summary

| Model | 추천도 | 사용 케이스 |
|-------|--------|------------|
| **DINOv2 + ML-Decoder** | ⭐⭐⭐⭐⭐ | 세밀한 객체 인식 (계단, 의자 유형 등) |
| **EVA-02 + ML-Decoder** | ⭐⭐⭐⭐⭐ | 강력한 MIM 기반 representation |
| **SAM-2 + ML-Decoder** | ⭐⭐⭐⭐ | Robust spatial features |
| **ConvNeXt V2 + ML-Decoder** | ⭐⭐⭐⭐ | 효율성과 성능의 균형 (FCMAE) |
| **ConvNeXt + ML-Decoder** | ⭐⭐⭐⭐ | 효율성과 성능의 균형 |
| **SigLIP2 + ML-Decoder** | ⭐⭐⭐⭐ | 범용성과 안정성 |
| **SigLIP + ML-Decoder** | ⭐⭐⭐ | 빠른 실험, 베이스라인 |

## 상세 비교표

### 1. 아키텍처 & 학습 방식

| Model | Type | Training Method | Pretraining Dataset |
|-------|------|-----------------|---------------------|
| **SigLIP** | Vision Transformer | Contrastive (Sigmoid) | WebLI |
| **SigLIP2** | Improved ViT | Contrastive (Sigmoid) | WebLI |
| **DINOv2** | Vision Transformer | Self-supervised (DINO) | LVD-142M |
| **ConvNeXt** | CNN | Supervised | ImageNet-22k |
| **ConvNeXt V2** | CNN | Self-supervised (FCMAE) | ImageNet-22k |
| **SAM-2** | Hiera (Hierarchical ViT) | Segmentation | SA-1B + Videos |
| **EVA-02** | Vision Transformer | Masked Image Modeling | ImageNet-22k |

### 2. 출력 구조 & Token Processing

| Model | Raw Output | Shape | Spatial Tokens | CLS Token | Processing |
|-------|-----------|-------|----------------|-----------|------------|
| **SigLIP** | 3D sequence | [B, 196, 768] | 196 (14×14) | ✗ | Use as-is |
| **SigLIP2** | 3D sequence | [B, 196, 768] | 196 (14×14) | ✗ | Use as-is |
| **DINOv2** | 3D sequence | [B, 257, 768] | 256 (16×16) | ✓ (removed) | Remove CLS |
| **ConvNeXt** | 4D feature map | [B, 1024, 7, 7] | 49 (7×7) | ✗ | Flatten |
| **ConvNeXt V2** | 4D feature map | [B, 1024, 7, 7] | 49 (7×7) | ✗ | Flatten |
| **SAM-2** | 4D feature map | [B, H, W, 896] | varies | ✗ | Reshape |
| **EVA-02** | 3D sequence | [B, 257, 768] | 256 (16×16) | ✓ (removed) | Remove CLS |

**ML-Decoder Input (after processing):**

```
SigLIP:     [batch, 196, 768]  ← 196 spatial locations
SigLIP2:    [batch, 196, 768]  ← 196 spatial locations
DINOv2:     [batch, 256, 768]  ← 256 spatial locations
ConvNeXt:   [batch, 49, 1024]  ← 49 spatial locations
ConvNeXt V2:[batch, 49, 1024]  ← 49 spatial locations
SAM-2:      [batch, H*W, 896]  ← varies spatial locations
EVA-02:     [batch, 256, 768]  ← 256 spatial locations
```

### 3. Classification Heads 비교

| Head | 파라미터 | 특징 | 적합한 케이스 |
|------|---------|------|--------------|
| **Linear** | ~4K | 간단, 빠름 | 빠른 실험, 베이스라인 |
| **ML-Decoder** | ~7.9M | Transformer decoder, Query embeddings | Multi-label, 세밀한 인식 |
| **CSRA** | ~15K | Class-specific attention + Residual | 경량, 효율적 |
| **Q2L** | ~16M | Label correlation 모델링 | 레이블 간 관계 중요 |

### 4. 성능 특성

#### 세밀한 객체 인식 (Fine-grained)

| Model | Rating | Details |
|-------|--------|---------|
| **DINOv2** | ⭐⭐⭐⭐⭐ | Self-supervised로 local features 강화, 256 locations |
| **EVA-02** | ⭐⭐⭐⭐⭐ | MIM으로 강력한 representation, 256 locations |
| **SAM-2** | ⭐⭐⭐⭐ | Segmentation 학습으로 spatial awareness 강화 |
| **SigLIP2** | ⭐⭐⭐⭐ | Improved contrastive learning, 196 locations |
| **SigLIP** | ⭐⭐⭐ | Baseline contrastive learning |
| **ConvNeXt V2** | ⭐⭐⭐ | FCMAE로 개선, 49 locations (coarse) |
| **ConvNeXt** | ⭐⭐⭐ | CNN inductive bias, 49 locations (coarse) |

#### 효율성

| Model | Training Speed | Memory | Inference |
|-------|---------------|--------|-----------|
| **ConvNeXt** | ⚡⚡⚡⚡ | Low | Fast |
| **ConvNeXt V2** | ⚡⚡⚡⚡ | Low | Fast |
| **SigLIP** | ⚡⚡⚡ | Medium | Medium |
| **SigLIP2** | ⚡⚡⚡ | Medium | Medium |
| **EVA-02** | ⚡⚡⚡ | Medium | Medium |
| **DINOv2** | ⚡⚡ | Medium-High | Medium |
| **SAM-2** | ⚡⚡ | High | Slower |

### 5. 클래스별 예상 성능

우리 태스크의 5개 클래스:

| Class | DINOv2 | EVA-02 | SAM-2 | ConvNeXt | SigLIP2 | 난이도 |
|-------|--------|--------|-------|----------|---------|--------|
| **has_step** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| **has_movable_chair** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Hard |
| **has_high_chair** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Hard |
| **has_fixed_chair** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Hard |
| **has_floor_chair** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |

## 추천 전략

### 시나리오 1: 최고 성능 필요
```bash
./run_training.sh --model-type dinov2-mldecoder
# 또는
./run_training.sh --model-type eva02-mldecoder
```

### 시나리오 2: 효율성 중요
```bash
./run_training.sh --model-type convnext-mldecoder
# 또는
./run_training.sh --model-type convnextv2-mldecoder
```

### 시나리오 3: Robust Spatial Features
```bash
./run_training.sh --model-type sam2-mldecoder
```

### 시나리오 4: 범용성 & 안정성
```bash
./run_training.sh --model-type siglip2-mldecoder
```

### 시나리오 5: 경량 Head로 빠른 실험
```bash
./run_training.sh --model-type dinov2-csra
./run_training.sh --model-type eva02-csra
```

## 사용 가능한 조합 (28가지)

```bash
# SigLIP 계열 (8가지)
siglip, siglip-mldecoder, siglip-csra, siglip-q2l
siglip2, siglip2-mldecoder, siglip2-csra, siglip2-q2l

# DINOv2 계열 (4가지)
dinov2, dinov2-mldecoder, dinov2-csra, dinov2-q2l

# ConvNeXt 계열 (8가지)
convnext, convnext-mldecoder, convnext-csra, convnext-q2l
convnextv2, convnextv2-mldecoder, convnextv2-csra, convnextv2-q2l

# SAM-2 계열 (4가지)
sam2, sam2-mldecoder, sam2-csra, sam2-q2l

# EVA-02 계열 (4가지)
eva02, eva02-mldecoder, eva02-csra, eva02-q2l
```

## 참고 자료

### Models
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [ConvNeXt V2 Paper](https://arxiv.org/abs/2301.00808)
- [SAM-2 Paper](https://arxiv.org/abs/2408.00714)
- [EVA-02 Paper](https://arxiv.org/abs/2303.11331)

### Classification Heads
- [ML-Decoder Paper](https://arxiv.org/abs/2111.12933)
- [CSRA Paper](https://arxiv.org/abs/2108.02456)
- [Q2L Paper](https://arxiv.org/abs/2107.10834)
