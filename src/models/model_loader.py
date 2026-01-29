import os
from typing import Optional

import torch
import torch.nn as nn
import yaml
from transformers import (
    SiglipForImageClassification,
    Dinov2ForImageClassification,
    ConvNextForImageClassification,
    ConvNextV2ForImageClassification,
    AutoModelForImageClassification,
    AutoModel,
    Sam2Model,
)
from transformers.modeling_outputs import ImageClassifierOutput

from .heads import MLDecoder, CSRA, Q2L


class Sam2ForImageClassification(nn.Module):
    """
    SAM-2 Vision Encoder wrapper for Image Classification.

    SAM-2 doesn't have a native ForImageClassification class,
    so we create a wrapper that extracts features and adds a classifier.
    """

    def __init__(self, model_id: str, num_labels: int):
        super().__init__()
        self.sam2 = Sam2Model.from_pretrained(model_id)
        self.vision_encoder = self.sam2.vision_encoder

        # Get hidden size from a forward pass
        import torch
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.vision_encoder(dummy)
            self.hidden_size = out.last_hidden_state.shape[-1]

        # Add classifier
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels

        # Create a config-like object for compatibility
        class Config:
            def __init__(self, hidden_size, num_labels):
                self.hidden_size = hidden_size
                self.num_labels = num_labels
        self.config = Config(self.hidden_size, num_labels)

    def forward(self, pixel_values=None, labels=None, return_dict=True, **kwargs):
        # Get vision encoder output
        vision_output = self.vision_encoder(pixel_values)

        # last_hidden_state: [batch, H, W, hidden_size] -> [batch, hidden_size]
        features = vision_output.last_hidden_state
        # Global average pooling
        pooled = features.mean(dim=[1, 2])

        # Classify
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


class Eva02ForImageClassification(nn.Module):
    """
    EVA-02 Vision Transformer wrapper for Image Classification.

    EVA-02 is loaded via timm wrapper and doesn't have a native
    ForImageClassification class, so we create a wrapper.
    """

    def __init__(self, model_id: str, num_labels: int):
        super().__init__()
        # Load EVA-02 via AutoModel (timm wrapper)
        self.eva02 = AutoModel.from_pretrained(model_id)
        self.hidden_size = 768  # EVA-02 base hidden size

        # Get actual hidden size from forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.eva02(dummy)
            self.hidden_size = out.last_hidden_state.shape[-1]

        # Add classifier
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels

        # Create a config-like object for compatibility
        class Config:
            def __init__(self, hidden_size, num_labels):
                self.hidden_size = hidden_size
                self.num_labels = num_labels
        self.config = Config(self.hidden_size, num_labels)

    def forward(self, pixel_values=None, labels=None, return_dict=True, **kwargs):
        # Get vision encoder output
        vision_output = self.eva02(pixel_values)

        # Use pooler_output (CLS token) for classification
        # Shape: [batch, hidden_size]
        pooled = vision_output.pooler_output

        # Classify
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODELS_DIR))
_DEFAULT_CONFIG_PATH = os.path.join(
    _PROJECT_ROOT, "models", "configs", "siglip_config.yaml"
)


class VisionMLDecoderModel(nn.Module):
    """
    Generic Vision model with ML Decoder that uses feature maps.

    This model extracts spatial feature maps from the vision encoder
    and passes them to the ML Decoder for multi-label classification.

    Supports: SigLIP, DINOv2, ConvNeXt, and other vision transformers.
    """

    def __init__(self, vision_model, classifier, model_type="siglip"):
        """
        Args:
            vision_model: Vision model (e.g., SigLIP, DINOv2, ConvNeXt)
            classifier: ML Decoder classifier
            model_type: Type of model ("siglip", "dinov2", "convnext")
        """
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier
        self.model_type = model_type

    def forward(self, pixel_values=None, labels=None, return_dict=True, **kwargs):
        """
        Forward pass using feature maps instead of pooled output.

        Args:
            pixel_values: Input images
            labels: Ground truth labels for loss computation
            return_dict: Whether to return dict or tuple

        Returns:
            ImageClassifierOutput with logits and optional loss
        """
        # Extract feature maps from vision encoder
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )

        # Get feature map (all spatial tokens)
        # Shape: [batch, num_patches, hidden_size]
        # - SigLIP: [batch, 196, 768] - NO CLS token, all are patch embeddings
        # - DINOv2: [batch, 257, 768] - CLS token (first) + 256 patch embeddings
        # - ConvNeXt: [batch, C, H, W] - CNN feature map, needs flatten
        # - SAM2: [batch, H, W, hidden_size] - Hiera output, needs reshape
        feature_map = vision_outputs.last_hidden_state

        # Handle 4D feature maps
        if feature_map.dim() == 4:
            if self.model_type == "sam2":
                # SAM-2 (Hiera): [batch, H, W, hidden_size] -> [batch, H*W, hidden_size]
                batch_size, h, w, hidden_size = feature_map.shape
                feature_map = feature_map.reshape(batch_size, h * w, hidden_size)
            else:
                # ConvNeXt: [batch, C, H, W] -> [batch, H*W, C]
                feature_map = feature_map.flatten(2).transpose(1, 2)

        # Remove CLS token for models that have it (DINOv2, EVA-02)
        # ML-Decoder works best with spatial features only
        if self.model_type in ("dinov2", "eva02"):
            # DINOv2/EVA-02: Remove CLS token (first token)
            # [batch, 257, 768] -> [batch, 256, 768]
            feature_map = feature_map[:, 1:, :]

        # SigLIP has no CLS token, all tokens are patch embeddings
        # ConvNeXt uses spatial feature maps; no CLS token
        # SAM-2 has no CLS token, uses Hiera spatial features

        # Pass to ML Decoder
        logits = self.classifier(feature_map)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _detect_model_type(model_id: str) -> str:
    """
    모델 ID로부터 모델 타입을 자동 감지합니다.

    Args:
        model_id: Hugging Face 모델 ID (e.g., "google/siglip-base-patch16-224")

    Returns:
        str: 모델 타입 ("siglip", "dinov2", "convnext", "convnextv2", "sam2", "eva02", "unknown")
    """
    model_id_lower = model_id.lower()

    if "siglip" in model_id_lower:
        return "siglip"
    elif "dinov2" in model_id_lower or "dino-v2" in model_id_lower:
        return "dinov2"
    elif "convnextv2" in model_id_lower or "convnext-v2" in model_id_lower:
        return "convnextv2"
    elif "convnext" in model_id_lower:
        return "convnext"
    elif "sam2" in model_id_lower:
        return "sam2"
    elif "eva02" in model_id_lower or "eva-02" in model_id_lower:
        return "eva02"
    else:
        return "unknown"


def _get_model_class(model_type: str):
    """
    모델 타입에 따라 적절한 ForImageClassification 클래스를 반환합니다.

    Args:
        model_type: 모델 타입 ("siglip", "dinov2", "convnext", "convnextv2", "sam2", "eva02")

    Returns:
        ForImageClassification 클래스 (sam2, eva02는 None 반환, 별도 처리)
    """
    if model_type == "siglip":
        return SiglipForImageClassification
    elif model_type == "dinov2":
        return Dinov2ForImageClassification
    elif model_type == "convnext":
        return ConvNextForImageClassification
    elif model_type == "convnextv2":
        return ConvNextV2ForImageClassification
    elif model_type == "sam2":
        return None  # SAM2 requires special handling
    elif model_type == "eva02":
        return None  # EVA-02 requires special handling (timm wrapper)
    else:
        # Fallback to AutoModel
        return AutoModelForImageClassification


def load_model(
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    freeze_vision_encoder: Optional[bool] = None,
    config_path: Optional[str] = None,
):
    """
    Vision 모델을 로드합니다 (SigLIP, DINOv2, ConvNeXt 등).

    Args:
        model_name (str, optional): 직접 지정할 모델 ID.
        num_labels (int, optional): 분류할 레이블 수.
        freeze_vision_encoder (bool, optional): Vision encoder 동결 여부.
        config_path (str, optional): YAML 설정 파일 경로.

    Returns:
        model: 로드된 Vision 분류 모델.

    Supported Models:
        - SigLIP: google/siglip-base-patch16-224, google/siglip2-base-patch16-224
        - DINOv2: facebook/dinov2-base, facebook/dinov2-large
        - ConvNeXt: facebook/convnext-base-224, facebook/convnext-base-224-22k
    """
    resolved_name = model_name
    resolved_num_labels = num_labels
    resolved_freeze = freeze_vision_encoder
    resolved_config_path = config_path
    config = None

    if resolved_config_path or resolved_name is None or resolved_num_labels is None or resolved_freeze is None:
        if resolved_config_path is None:
            resolved_config_path = _DEFAULT_CONFIG_PATH
        config = _load_config(resolved_config_path)

        if resolved_name is None:
            resolved_name = config.get("model", {}).get("pretrained_model_id")
        if resolved_num_labels is None:
            resolved_num_labels = config.get("model", {}).get("num_labels")
        if resolved_freeze is None:
            resolved_freeze = config.get("training", {}).get("freeze_vision_encoder", False)

    if resolved_name is None:
        raise ValueError("model_name을 직접 지정하거나 config에 model.pretrained_model_id를 설정해야 합니다.")
    if resolved_num_labels is None:
        raise ValueError("num_labels를 직접 지정하거나 config에 model.num_labels를 설정해야 합니다.")

    # 모델 타입 자동 감지
    model_type = _detect_model_type(resolved_name)
    model_class = _get_model_class(model_type)

    config_info = f" (config: {resolved_config_path})" if resolved_config_path else ""
    print(f"'{resolved_name}' 모델을 로드합니다...{config_info}")
    print(f"모델 타입: {model_type}")
    print(f"레이블 수: {resolved_num_labels}")

    try:
        # SAM-2 and EVA-02 require special handling (no native ForImageClassification)
        if model_type == "sam2":
            model = Sam2ForImageClassification(
                model_id=resolved_name,
                num_labels=resolved_num_labels
            )
            print(f"✓ Sam2ForImageClassification 로드 완료 (custom wrapper)")
        elif model_type == "eva02":
            model = Eva02ForImageClassification(
                model_id=resolved_name,
                num_labels=resolved_num_labels
            )
            print(f"✓ Eva02ForImageClassification 로드 완료 (timm wrapper)")
        else:
            model = model_class.from_pretrained(
                resolved_name,
                num_labels=resolved_num_labels,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True  # 분류 헤드 크기가 다를 수 있음
            )
            print(f"✓ {model_class.__name__} 로드 완료")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise

    # Classification head 타입 결정 (ML Decoder, CSRA, Q2L, or Linear)
    use_ml_decoder = False
    use_csra = False
    use_q2l = False

    if config is not None:
        ml_decoder_config = config.get("ml_decoder", {})
        csra_config = config.get("csra", {})
        q2l_config = config.get("q2l", {})

        use_ml_decoder = ml_decoder_config.get("enabled", False)
        use_csra = csra_config.get("enabled", False)
        use_q2l = q2l_config.get("enabled", False)

        # 여러 head가 동시에 활성화된 경우 오류
        enabled_heads = [name for name, enabled in [
            ("ml_decoder", use_ml_decoder),
            ("csra", use_csra),
            ("q2l", use_q2l)
        ] if enabled]

        if len(enabled_heads) > 1:
            raise ValueError(f"여러 head를 동시에 활성화할 수 없습니다: {enabled_heads}. 하나만 선택하세요.")

        if use_ml_decoder:
            print("\n=== ML Decoder 적용 (Feature Map 버전) ===")

            # Extract vision model based on model type
            if model_type == "siglip":
                vision_model = model.vision_model
                num_patches = 196  # 14x14 for 224x224 image with patch size 16
                use_cls_token = False
            elif model_type == "dinov2":
                vision_model = model.dinov2
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
                use_cls_token = True  # Has CLS token but we remove it
            elif model_type == "convnext":
                vision_model = model.convnext
                num_patches = "varies"  # Depends on architecture
                use_cls_token = False
            elif model_type == "convnextv2":
                vision_model = model.convnextv2
                num_patches = "varies"  # Depends on architecture
                use_cls_token = False
            elif model_type == "sam2":
                vision_model = model.vision_encoder
                num_patches = "varies"  # Depends on SAM2 architecture (Hiera)
                use_cls_token = False
            elif model_type == "eva02":
                vision_model = model.eva02
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
                use_cls_token = True  # Has CLS token but we remove it
            else:
                # Try common attribute names
                if hasattr(model, 'vision_model'):
                    vision_model = model.vision_model
                elif hasattr(model, 'base_model'):
                    vision_model = model.base_model
                else:
                    raise ValueError(f"Unknown vision model extraction for model type: {model_type}")
                num_patches = "unknown"
                use_cls_token = False

            # Get hidden size from vision model config (handles SigLIP2/ConvNeXt)
            hidden_size = None
            # SAM-2, EVA-02: use hidden_size from wrapper class
            if model_type in ("sam2", "eva02"):
                hidden_size = model.hidden_size
            else:
                vision_cfg = getattr(vision_model, "config", None)
                if vision_cfg is not None:
                    hidden_size = getattr(vision_cfg, "hidden_size", None)
                    if hidden_size is None:
                        hidden_sizes = getattr(vision_cfg, "hidden_sizes", None)
                        if hidden_sizes:
                            hidden_size = hidden_sizes[-1]
                if hidden_size is None:
                    hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("hidden_size를 찾을 수 없습니다. vision model config를 확인하세요.")

            # Create ML Decoder
            ml_decoder = MLDecoder(
                num_classes=resolved_num_labels,
                initial_num_features=hidden_size,
                num_queries=ml_decoder_config.get("num_queries", resolved_num_labels),
                num_layers=ml_decoder_config.get("num_layers", 1),
                num_heads=ml_decoder_config.get("num_heads", 8),
                dim_feedforward=ml_decoder_config.get("dim_feedforward", 2048),
                dropout=ml_decoder_config.get("dropout", 0.1),
                normalize_before=ml_decoder_config.get("normalize_before", False),
                return_intermediate=ml_decoder_config.get("return_intermediate", False),
            )

            print(f"ML Decoder 생성 완료:")
            print(f"  Model type: {model_type}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Spatial patches: {num_patches}")
            if use_cls_token:
                print(f"  CLS token: Present but removed (spatial features only)")
            print(f"  Num queries: {ml_decoder_config.get('num_queries', resolved_num_labels)}")
            print(f"  Num layers: {ml_decoder_config.get('num_layers', 1)}")
            print(f"  Num heads: {ml_decoder_config.get('num_heads', 8)}")

            # Calculate ML Decoder parameters
            ml_decoder_params = sum(p.numel() for p in ml_decoder.parameters())
            print(f"  ML Decoder parameters: {ml_decoder_params:,}")

            # Create custom model that uses feature maps
            model = VisionMLDecoderModel(
                vision_model=vision_model,
                classifier=ml_decoder,
                model_type=model_type
            )
            print(f"✓ Feature map을 사용하는 VisionMLDecoderModel로 교체 완료")
            print(f"  Input shape: [batch, {num_patches}, {hidden_size}] (spatial tokens only)")
            print(f"  Cross-attention: Each query attends to {num_patches} spatial locations")

        elif use_csra:
            print("\n=== CSRA Head 적용 ===")

            # Extract vision model based on model type
            if model_type == "siglip":
                vision_model = model.vision_model
                num_patches = 196  # 14x14 for 224x224 image with patch size 16
            elif model_type == "dinov2":
                vision_model = model.dinov2
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
            elif model_type == "convnext":
                vision_model = model.convnext
                num_patches = 49  # 7x7 for 224x224 image
            elif model_type == "convnextv2":
                vision_model = model.convnextv2
                num_patches = 49  # 7x7 for 224x224 image
            elif model_type == "sam2":
                vision_model = model.vision_encoder
                num_patches = "varies"  # Depends on SAM2 architecture (Hiera)
            elif model_type == "eva02":
                vision_model = model.eva02
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
            else:
                if hasattr(model, 'vision_model'):
                    vision_model = model.vision_model
                elif hasattr(model, 'base_model'):
                    vision_model = model.base_model
                else:
                    raise ValueError(f"Unknown vision model extraction for model type: {model_type}")
                num_patches = "unknown"

            # Get hidden size from vision model config
            hidden_size = None
            # SAM-2, EVA-02: use hidden_size from wrapper class
            if model_type in ("sam2", "eva02"):
                hidden_size = model.hidden_size
            else:
                vision_cfg = getattr(vision_model, "config", None)
                if vision_cfg is not None:
                    hidden_size = getattr(vision_cfg, "hidden_size", None)
                    if hidden_size is None:
                        hidden_sizes = getattr(vision_cfg, "hidden_sizes", None)
                        if hidden_sizes:
                            hidden_size = hidden_sizes[-1]
                if hidden_size is None:
                    hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("hidden_size를 찾을 수 없습니다. vision model config를 확인하세요.")

            # Create CSRA Head
            csra_head = CSRA(
                input_dim=hidden_size,
                num_classes=resolved_num_labels,
                num_heads=csra_config.get("num_heads", 4),
                lam=csra_config.get("lam", 0.1),
                temperatures=csra_config.get("temperatures", None),
                dropout=csra_config.get("dropout", 0.1),
            )

            print(f"CSRA Head 생성 완료:")
            print(f"  Model type: {model_type}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Spatial patches: {num_patches}")
            print(f"  Num heads: {csra_config.get('num_heads', 4)}")
            print(f"  Lambda (residual weight): {csra_config.get('lam', 0.1)}")
            print(f"  Temperatures: {csra_head.temperatures}")

            # Calculate CSRA parameters
            csra_params = sum(p.numel() for p in csra_head.parameters())
            print(f"  CSRA parameters: {csra_params:,}")

            # Create custom model that uses feature maps
            model = VisionMLDecoderModel(
                vision_model=vision_model,
                classifier=csra_head,
                model_type=model_type
            )
            print(f"✓ CSRA Head를 사용하는 VisionMLDecoderModel로 교체 완료")

        elif use_q2l:
            print("\n=== Q2L (Query2Label) Head 적용 ===")

            # Extract vision model based on model type
            if model_type == "siglip":
                vision_model = model.vision_model
                num_patches = 196  # 14x14 for 224x224 image with patch size 16
            elif model_type == "dinov2":
                vision_model = model.dinov2
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
            elif model_type == "convnext":
                vision_model = model.convnext
                num_patches = 49  # 7x7 for 224x224 image
            elif model_type == "convnextv2":
                vision_model = model.convnextv2
                num_patches = 49  # 7x7 for 224x224 image
            elif model_type == "sam2":
                vision_model = model.vision_encoder
                num_patches = "varies"  # Depends on SAM2 architecture (Hiera)
            elif model_type == "eva02":
                vision_model = model.eva02
                num_patches = 256  # 16x16 patches (CLS token will be removed in forward)
            else:
                if hasattr(model, 'vision_model'):
                    vision_model = model.vision_model
                elif hasattr(model, 'base_model'):
                    vision_model = model.base_model
                else:
                    raise ValueError(f"Unknown vision model extraction for model type: {model_type}")
                num_patches = "unknown"

            # Get hidden size from vision model config
            hidden_size = None
            # SAM-2, EVA-02: use hidden_size from wrapper class
            if model_type in ("sam2", "eva02"):
                hidden_size = model.hidden_size
            else:
                vision_cfg = getattr(vision_model, "config", None)
                if vision_cfg is not None:
                    hidden_size = getattr(vision_cfg, "hidden_size", None)
                    if hidden_size is None:
                        hidden_sizes = getattr(vision_cfg, "hidden_sizes", None)
                        if hidden_sizes:
                            hidden_size = hidden_sizes[-1]
                if hidden_size is None:
                    hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("hidden_size를 찾을 수 없습니다. vision model config를 확인하세요.")

            # Create Q2L Head
            q2l_head = Q2L(
                input_dim=hidden_size,
                num_classes=resolved_num_labels,
                num_layers=q2l_config.get("num_layers", 2),
                num_heads=q2l_config.get("num_heads", 8),
                dim_feedforward=q2l_config.get("dim_feedforward", 2048),
                dropout=q2l_config.get("dropout", 0.1),
                use_pos_encoding=q2l_config.get("use_pos_encoding", True),
            )

            print(f"Q2L Head 생성 완료:")
            print(f"  Model type: {model_type}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Spatial patches: {num_patches}")
            print(f"  Num layers: {q2l_config.get('num_layers', 2)}")
            print(f"  Num heads: {q2l_config.get('num_heads', 8)}")
            print(f"  Dim feedforward: {q2l_config.get('dim_feedforward', 2048)}")
            print(f"  Use pos encoding: {q2l_config.get('use_pos_encoding', True)}")

            # Calculate Q2L parameters
            q2l_params = sum(p.numel() for p in q2l_head.parameters())
            print(f"  Q2L parameters: {q2l_params:,}")

            # Create custom model that uses feature maps
            model = VisionMLDecoderModel(
                vision_model=vision_model,
                classifier=q2l_head,
                model_type=model_type
            )
            print(f"✓ Q2L Head를 사용하는 VisionMLDecoderModel로 교체 완료")

    if resolved_freeze:
        print("모델의 Vision Encoder를 동결합니다...")
        for name, param in model.named_parameters():
            if name.startswith("vision_model"):
                param.requires_grad = False
        print("Vision Encoder 동결 완료.")
    else:
        print("freeze_vision_encoder=False: Vision Encoder를 동결하지 않습니다.")

    return model


if __name__ == '__main__':
    # 테스트 코드
    print("=== SigLIP 모델 로드 테스트 ===")

    # Config 없이 직접 로드
    model = load_model(
        model_name="google/siglip-base-patch16-224",
        num_labels=5,
        freeze_vision_encoder=True
    )

    print(f"모델 로드 성공!")
    print(f"모델 타입: {type(model)}")
