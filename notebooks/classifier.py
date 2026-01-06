import os
import sys
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path

# Add src to path for imports
NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 레이블 순서 (dataset.py와 동일)
LABEL_NAMES = [
    "has_step",
    "has_movable_chair",
    "has_high_chair",
    "has_fixed_chair",
    "has_floor_chair"
]


class SigLIPClassifier:
    """
    SigLIP 모델을 사용한 Multi-label Classification
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        checkpoint_path: Optional[str] = None,
        threshold: float = 0.5,
        label_names: Optional[List[str]] = None
    ):
        """
        SigLIPClassifier 초기화

        Args:
            model_id (str): Hugging Face 모델 ID
            checkpoint_path (str, optional): 파인튜닝된 모델 체크포인트 경로
            threshold (float): Sigmoid threshold for prediction
            label_names (List[str], optional): 레이블 이름 리스트
        """
        print("모델 로딩 중...")
        self.model_id = model_id
        self.threshold = threshold
        self.label_names = label_names or LABEL_NAMES

        # Processor 로드
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

        # 모델 로드
        num_labels = len(self.label_names)

        # 체크포인트가 있는 경우, config를 읽어서 ML Decoder 사용 여부 확인
        if checkpoint_path:
            print(f"파인튜닝된 가중치 로딩: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # checkpoint에서 config 읽기
            saved_config = checkpoint.get('config', None)

            if saved_config and saved_config.get('ml_decoder', {}).get('enabled', False):
                # ML Decoder 모델 사용
                print("체크포인트에서 ML Decoder 설정을 감지했습니다.")
                from models.model_loader import load_model
                import tempfile
                import yaml

                # 임시 config 파일 생성
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(saved_config, f)
                    temp_config_path = f.name

                try:
                    self.model = load_model(config_path=temp_config_path)
                finally:
                    os.unlink(temp_config_path)
            else:
                # 일반 SigLIP 모델
                self.model = SiglipForImageClassification.from_pretrained(
                    model_id,
                    num_labels=num_labels,
                    problem_type="multi_label_classification",
                    ignore_mismatched_sizes=True
                )

            # 가중치 로드
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"경고: {len(missing_keys)}개의 키가 체크포인트에 없습니다.")
            if unexpected_keys:
                print(f"경고: {len(unexpected_keys)}개의 예상치 못한 키가 체크포인트에 있습니다.")
            print("모델 가중치 로드 완료.")
        else:
            # Pretrained 모델만 사용
            self.model = SiglipForImageClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True
            )

        # GPU 또는 MPS 사용
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()
        print(f"모델이 {self.device} 디바이스로 이동되었습니다.")

    def classify(
        self,
        image_path: str,
        threshold: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        이미지에서 multi-label classification 수행

        Args:
            image_path (str): 이미지 파일 경로
            threshold (float, optional): Sigmoid threshold (None이면 기본값 사용)

        Returns:
            Dict[str, bool]: 각 레이블별 예측 결과
        """
        if threshold is None:
            threshold = self.threshold

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Sigmoid + threshold
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > threshold).astype(int)

        # 결과 딕셔너리 생성
        result = {}
        for i, label_name in enumerate(self.label_names):
            result[label_name] = bool(preds[i])
            result[f"{label_name}_prob"] = float(probs[i])

        return result

    def batch_classify(
        self,
        image_paths: List[str],
        threshold: Optional[float] = None
    ) -> List[Dict[str, bool]]:
        """
        여러 이미지에 대해 일괄 분류

        Args:
            image_paths (List[str]): 이미지 파일 경로 리스트
            threshold (float, optional): Sigmoid threshold

        Returns:
            List[Dict[str, bool]]: 각 이미지의 분류 결과 리스트
        """
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"처리 중: {i+1}/{len(image_paths)} - {os.path.basename(img_path)}")
            result = self.classify(img_path, threshold)
            result["image_path"] = img_path
            results.append(result)
        return results


if __name__ == "__main__":
    # 테스트 코드
    import argparse

    parser = argparse.ArgumentParser(description="SigLIP Multi-Label Classification")
    parser.add_argument("--image", type=str, required=True, help="이미지 파일 경로")
    parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold")
    parser.add_argument("--model-id", type=str, default="google/siglip-base-patch16-224", help="모델 ID")

    args = parser.parse_args()

    print("=" * 60)
    print("SigLIP Multi-Label Classification")
    print("=" * 60)
    print(f"모델 ID: {args.model_id}")
    print(f"체크포인트: {args.checkpoint or '없음 (기본 모델)'}")
    print(f"임계값: {args.threshold}")
    print("=" * 60)
    print()

    # 분류 실행
    classifier = SigLIPClassifier(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold
    )

    result = classifier.classify(image_path=args.image)

    print("\n" + "=" * 60)
    print("분류 결과:")
    print("=" * 60)
    for key, value in result.items():
        if not key.endswith("_prob"):
            status = "✓" if value else "✗"
            prob = result.get(f"{key}_prob", 0.0)
            print(f"  {status} {key}: {value} (확률: {prob:.3f})")
    print("=" * 60)
