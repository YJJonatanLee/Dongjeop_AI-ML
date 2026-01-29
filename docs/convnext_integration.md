# CNN Models Integration (ConvNeXt / ConvNeXt V2)

ConvNeXt와 ConvNeXt V2는 **CNN 기반** 모델로, ViT와는 다른 출력 구조를 가지고 있습니다. Classification heads와 통합하기 위해 추가 처리가 필요합니다.

## ConvNeXt vs ConvNeXt V2

| 항목 | ConvNeXt | ConvNeXt V2 |
|------|----------|-------------|
| 학습 방식 | Supervised | Self-supervised (FCMAE) |
| Model ID | `facebook/convnext-base-224-22k` | `facebook/convnextv2-base-1k-224` |
| 특징 | Modern CNN baseline | Masked autoencoder로 개선된 representation |
| Hidden size | 1024 | 1024 |

## 아키텍처

```
Input: [B, 3, 224, 224]
    ↓ Patch embedding (4×4 conv, stride 4)
[B, C, 56, 56]
    ↓ Stage 0
[B, C, 56, 56]
    ↓ Stage 1 (downsample)
[B, C, 28, 28]
    ↓ Stage 2 (downsample)
[B, C, 14, 14]
    ↓ Stage 3 (downsample)
[B, C, 7, 7]  ← Final feature map
```

**ConvNeXt-base:**
- Channels (C): **1024**
- Final spatial resolution: **7×7**
- Total spatial locations: **49**

## ViT vs CNN 출력 비교

| Model | Output Structure | Shape | Spatial Locations |
|-------|-----------------|-------|-------------------|
| **DINOv2** | Token sequence (3D) | [B, 257, 768] | 256 (16×16) |
| **SigLIP** | Token sequence (3D) | [B, 196, 768] | 196 (14×14) |
| **EVA-02** | Token sequence (3D) | [B, 257, 768] | 256 (16×16) |
| **SAM-2** | Feature map (4D) | [B, H, W, 896] | varies |
| **ConvNeXt** | Feature map (4D) | [B, 1024, 7, 7] | 49 (7×7) |
| **ConvNeXt V2** | Feature map (4D) | [B, 1024, 7, 7] | 49 (7×7) |

**핵심 차이:**
- ViT: 이미 토큰 형태 (3D tensor)
- CNN: Feature map (4D tensor) → **변환 필요**

## Classification Head 입력 요구사항

Classification heads (ML-Decoder, CSRA, Q2L)는 **3D token sequence**를 입력으로 받습니다:
```
Input: [batch, seq_len, hidden_size]
```

따라서 CNN의 4D feature map을 3D로 변환해야 합니다.

## 변환 프로세스

### 1. CNN 출력
```python
vision_outputs = model.convnext(pixel_values=pixel_values, return_dict=True)
feature_map = vision_outputs.last_hidden_state
# Shape: [batch, channels, height, width]
# Example: [2, 1024, 7, 7]
```

### 2. Flatten 연산
```python
if feature_map.dim() == 4:
    # [B, C, H, W] → [B, C, H*W]
    feature_map = feature_map.flatten(2)
    # Example: [2, 1024, 49]

    # [B, C, H*W] → [B, H*W, C]
    feature_map = feature_map.transpose(1, 2)
    # Example: [2, 49, 1024]
```

### 3. Classification Head 입력
```python
logits = self.classifier(feature_map)
# Input: [batch, 49, 1024]
# Output: [batch, 5]
```

## 사용 방법

```bash
# ConvNeXt + Linear
./run_training.sh --model-type convnext

# ConvNeXt + ML-Decoder
./run_training.sh --model-type convnext-mldecoder

# ConvNeXt + CSRA
./run_training.sh --model-type convnext-csra

# ConvNeXt + Q2L
./run_training.sh --model-type convnext-q2l

# ConvNeXt V2 (동일한 패턴)
./run_training.sh --model-type convnextv2-mldecoder
./run_training.sh --model-type convnextv2-csra
./run_training.sh --model-type convnextv2-q2l
```

## Spatial Locations 비교

Classification heads가 attend하는 spatial locations 수:

| Model | Spatial Locations | Grid Size | Notes |
|-------|------------------|-----------|-------|
| DINOv2 | **256** | 16×16 | 높은 해상도 |
| EVA-02 | **256** | 16×16 | 높은 해상도 |
| SigLIP | **196** | 14×14 | 중간 해상도 |
| SAM-2 | **varies** | varies | 모델에 따라 다름 |
| ConvNeXt | **49** | 7×7 | 낮은 해상도 |
| ConvNeXt V2 | **49** | 7×7 | 낮은 해상도 |

**영향:**
- ConvNeXt는 적은 spatial locations (49)
- 더 coarse-grained attention
- 큰 객체나 전체적인 패턴에 유리
- 세밀한 디테일은 DINOv2/EVA-02가 더 유리

## 장단점

### 장점 ✅
1. **효율성**: CNN 기반으로 메모리/계산 효율적
2. **Inductive bias**: Convolution이 spatial structure 잘 캡처
3. **Translation invariance**: 위치 변화에 robust
4. **빠른 학습/추론**: ViT 대비 빠름

### 단점 ❌
1. **낮은 spatial resolution**: 49 locations (vs 196~256)
2. **세밀한 객체 인식**: 작은 객체나 디테일은 덜 정확할 수 있음

## 적합한 사용 케이스

ConvNeXt/ConvNeXt V2가 적합한 경우:
- ✅ 큰 객체 인식 (의자, 계단 전체)
- ✅ 전체적인 공간 구조 파악
- ✅ 효율성이 중요한 경우
- ✅ 실시간 추론 필요

DINOv2/EVA-02가 더 적합한 경우:
- ✅ 세밀한 객체 디테일 (의자 유형 구분)
- ✅ 작은 객체 인식
- ✅ Dense spatial information 필요

## 참고 자료

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [ConvNeXt V2 Paper](https://arxiv.org/abs/2301.00808)
- [ML-Decoder Paper](https://arxiv.org/abs/2111.12933)
