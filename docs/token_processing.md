# Token Processing for Classification Heads

Classification heads (ML-Decoder, CSRA, Q2L)는 spatial feature maps에 attention을 수행합니다. 각 모델마다 토큰 구조가 다르므로, 올바른 전처리가 필요합니다.

## 모델별 토큰 구조

### 1. SigLIP / SigLIP2

**출력 구조:**
```
Raw output: [batch, 196, 768]
└─ [0:196]: 196 patch tokens (14×14 spatial grid)
```

**Head 입력:**
```python
# CLS token 없음, 그대로 사용
feature_map = vision_outputs.last_hidden_state
# Result: [batch, 196, 768] - all spatial tokens
```

### 2. DINOv2

**출력 구조:**
```
Raw output: [batch, 257, 768]
├─ [0]: CLS token (global image representation)
└─ [1:257]: 256 patch tokens (16×16 spatial grid)
```

**Head 입력:**
```python
# CLS token 제거
feature_map = vision_outputs.last_hidden_state[:, 1:, :]
# Result: [batch, 256, 768] - spatial tokens only
```

### 3. ConvNeXt / ConvNeXt V2

**출력 구조:**
```
Raw output: [batch, 1024, 7, 7]
└─ 4D CNN feature map
```

**Head 입력:**
```python
# 4D → 3D 변환
feature_map = vision_outputs.last_hidden_state
feature_map = feature_map.flatten(2).transpose(1, 2)
# Result: [batch, 49, 1024] - 7×7 spatial locations
```

### 4. SAM-2

**출력 구조:**
```
Raw output: [batch, H, W, 896]
└─ 4D Hiera feature map (channels last)
```

**Head 입력:**
```python
# 4D → 3D 변환 (channels last format)
feature_map = vision_outputs.last_hidden_state
batch_size, h, w, hidden_size = feature_map.shape
feature_map = feature_map.reshape(batch_size, h * w, hidden_size)
# Result: [batch, H*W, 896] - spatial tokens
```

### 5. EVA-02

**출력 구조:**
```
Raw output: [batch, 257, 768]
├─ [0]: CLS token
└─ [1:257]: 256 patch tokens (16×16 spatial grid)
```

**Head 입력:**
```python
# CLS token 제거 (DINOv2와 동일)
feature_map = vision_outputs.last_hidden_state[:, 1:, :]
# Result: [batch, 256, 768] - spatial tokens only
```

## 요약 테이블

| Model | Raw Shape | CLS Token | Processing | Head Input |
|-------|-----------|-----------|------------|------------|
| SigLIP | [B, 196, 768] | ✗ | None | [B, 196, 768] |
| SigLIP2 | [B, 196, 768] | ✗ | None | [B, 196, 768] |
| DINOv2 | [B, 257, 768] | ✓ | Remove CLS | [B, 256, 768] |
| ConvNeXt | [B, 1024, 7, 7] | ✗ | Flatten | [B, 49, 1024] |
| ConvNeXt V2 | [B, 1024, 7, 7] | ✗ | Flatten | [B, 49, 1024] |
| SAM-2 | [B, H, W, 896] | ✗ | Reshape | [B, H*W, 896] |
| EVA-02 | [B, 257, 768] | ✓ | Remove CLS | [B, 256, 768] |

## VisionMLDecoderModel 구현

```python
def forward(self, pixel_values=None, labels=None, return_dict=True, **kwargs):
    # Get vision encoder outputs
    vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
    feature_map = vision_outputs.last_hidden_state

    # Handle 4D feature maps (ConvNeXt, SAM-2)
    if feature_map.dim() == 4:
        if self.model_type == "sam2":
            # SAM-2: [B, H, W, C] → [B, H*W, C]
            batch_size, h, w, hidden_size = feature_map.shape
            feature_map = feature_map.reshape(batch_size, h * w, hidden_size)
        else:
            # ConvNeXt: [B, C, H, W] → [B, H*W, C]
            feature_map = feature_map.flatten(2).transpose(1, 2)

    # Remove CLS token for models that have it
    if self.model_type in ("dinov2", "eva02"):
        feature_map = feature_map[:, 1:, :]

    # Pass to classifier head (ML-Decoder, CSRA, Q2L)
    logits = self.classifier(feature_map)
    return ImageClassifierOutput(loss=loss, logits=logits)
```

## Classification Heads

### ML-Decoder
- Query embeddings가 모든 spatial locations에 cross-attention
- 각 클래스별 query가 relevant 위치에 집중

### CSRA (Class-Specific Residual Attention)
- 클래스별 attention weights 계산
- Residual connection으로 안정적 학습
- 경량 (~15K params)

### Q2L (Query2Label)
- Transformer decoder 구조
- Label correlation 모델링
- 레이블 간 관계가 중요한 경우 유용

## 참고 자료

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ML-Decoder Paper](https://arxiv.org/abs/2111.12933)
- [CSRA Paper](https://arxiv.org/abs/2108.02456)
- [Q2L Paper](https://arxiv.org/abs/2107.10834)
