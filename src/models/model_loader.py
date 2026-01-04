import os
from typing import Optional

import yaml
from transformers import SiglipForImageClassification

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODELS_DIR))
_DEFAULT_CONFIG_PATH = os.path.join(
    _PROJECT_ROOT, "models", "configs", "siglip_config.yaml"
)


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    freeze_vision_encoder: Optional[bool] = None,
    config_path: Optional[str] = None,
):
    """
    SigLIP 모델을 로드합니다.

    Args:
        model_name (str, optional): 직접 지정할 모델 ID.
        num_labels (int, optional): 분류할 레이블 수.
        freeze_vision_encoder (bool, optional): Vision encoder 동결 여부.
        config_path (str, optional): YAML 설정 파일 경로.

    Returns:
        model: 로드된 SigLIP 분류 모델.
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

    config_info = f" (config: {resolved_config_path})" if resolved_config_path else ""
    print(f"'{resolved_name}' 모델을 로드합니다...{config_info}")
    print(f"레이블 수: {resolved_num_labels}")

    try:
        model = SiglipForImageClassification.from_pretrained(
            resolved_name,
            num_labels=resolved_num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True  # 분류 헤드 크기가 다를 수 있음
        )
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise

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
