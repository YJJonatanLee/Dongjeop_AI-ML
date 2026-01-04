"""
SigLIP Multi-Label Classification Inference Script

이미지 URL 또는 로컬 경로를 받아서 multi-label classification 결과를 반환합니다.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Union, Optional, List
from io import BytesIO

import requests
from PIL import Image

# 같은 폴더의 classifier 모듈 import
from classifier import SigLIPClassifier, LABEL_NAMES

# 프로젝트 루트 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


class SigLIPInference:
    """
    SigLIP 모델을 사용한 Multi-Label Classification 추론 클래스

    모델을 한 번만 로드하고 여러 이미지에 대해 추론을 수행할 수 있습니다.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        checkpoint_path: Optional[str] = None,
        threshold: float = 0.5,
        label_names: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        SigLIPInference 초기화

        Args:
            model_id (str): Hugging Face 모델 ID
                - "google/siglip-base-patch16-224" (SigLIP)
                - "google/siglip2-base-patch16-224" (SigLIP2)
            checkpoint_path (str, optional): 파인튜닝된 모델 체크포인트 경로
            threshold (float): Sigmoid threshold for prediction (기본값: 0.5)
            label_names (List[str], optional): 레이블 이름 리스트 (기본값: LABEL_NAMES)
            verbose (bool): 출력 메시지 표시 여부

        Examples:
            >>> # 기본 SigLIP 모델 사용
            >>> inferencer = SigLIPInference()

            >>> # 파인튜닝된 모델 사용
            >>> inferencer = SigLIPInference(
            ...     checkpoint_path="models/checkpoints/best_model.pth",
            ...     threshold=0.6
            ... )

            >>> # SigLIP2 사용
            >>> inferencer = SigLIPInference(
            ...     model_id="google/siglip2-base-patch16-224",
            ...     checkpoint_path="models/checkpoints/best_model.pth"
            ... )
        """
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.label_names = label_names or LABEL_NAMES
        self.verbose = verbose

        # SigLIPClassifier 초기화
        if self.verbose:
            print("모델 로딩 중...")

        self.classifier = SigLIPClassifier(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            label_names=self.label_names
        )

        if self.verbose:
            print("모델 로딩 완료!")
            print(f"레이블: {self.label_names}\n")

    @staticmethod
    def download_image_from_url(url: str, save_dir: Optional[str] = None) -> str:
        """
        URL에서 이미지를 다운로드하여 임시 파일로 저장합니다.

        Args:
            url (str): 이미지 URL
            save_dir (str, optional): 저장할 디렉토리. None이면 임시 디렉토리 사용

        Returns:
            str: 저장된 이미지 파일 경로
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 이미지 형식 확인
            image = Image.open(BytesIO(response.content))

            # 저장 디렉토리 설정
            if save_dir is None:
                save_dir = tempfile.gettempdir()

            # 파일 확장자 추출 (기본값은 .jpg)
            ext = image.format.lower() if image.format else "jpg"

            # 임시 파일 생성
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{ext}",
                dir=save_dir
            )

            # 이미지 저장
            image.save(temp_file.name)
            temp_file.close()

            return temp_file.name

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"이미지 다운로드 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"이미지 처리 실패: {e}")

    def infer(
        self,
        image_source: str,
        threshold: Optional[float] = None,
        cleanup: bool = True
    ) -> Dict[str, Union[bool, float]]:
        """
        단일 이미지에서 multi-label classification을 수행합니다.

        Args:
            image_source (str): 이미지 URL 또는 로컬 파일 경로
            threshold (float, optional): Sigmoid threshold (None이면 초기화 시 설정한 값 사용)
            cleanup (bool): URL에서 다운로드한 임시 파일 삭제 여부

        Returns:
            Dict[str, Union[bool, float]]: 각 레이블별 예측 결과 및 확률
                예: {
                    "has_step": True,
                    "has_step_prob": 0.876,
                    "has_movable_chair": False,
                    "has_movable_chair_prob": 0.234,
                    ...
                }

        Examples:
            >>> inferencer = SigLIPInference(checkpoint_path="model.pth")
            >>> result = inferencer.infer("./test.jpg")
            >>> print(result)
            {'has_step': True, 'has_step_prob': 0.876, ...}

            >>> # URL 이미지
            >>> result = inferencer.infer("https://example.com/image.jpg", threshold=0.6)
        """
        # 이미지 소스가 URL인지 로컬 파일인지 확인
        is_url = image_source.startswith(('http://', 'https://'))
        temp_file = None

        try:
            # URL인 경우 이미지 다운로드
            if is_url:
                if self.verbose:
                    print(f"URL에서 이미지 다운로드 중: {image_source}")
                image_path = self.download_image_from_url(image_source)
                temp_file = image_path
                if self.verbose:
                    print(f"이미지 다운로드 완료: {temp_file}\n")
            else:
                # 로컬 파일 경로 확인
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_source}")
                image_path = image_source

            # Multi-label classification 수행
            if self.verbose:
                print(f"분류 수행 중 (threshold={threshold or self.threshold})...")

            result = self.classifier.classify(
                image_path=image_path,
                threshold=threshold
            )

            if self.verbose:
                print(f"\n분류 결과:")
                for key, value in result.items():
                    if not key.endswith('_prob'):
                        prob = result.get(f"{key}_prob", 0.0)
                        status = "✓" if value else "✗"
                        print(f"  {status} {key}: {value} (확률: {prob:.3f})")

            return result

        finally:
            # 임시 파일 정리
            if cleanup and temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if self.verbose:
                        print(f"\n임시 파일 삭제: {temp_file}")
                except Exception as e:
                    if self.verbose:
                        print(f"임시 파일 삭제 실패: {e}")

    def batch_infer(
        self,
        image_sources: List[str],
        threshold: Optional[float] = None,
        cleanup: bool = True
    ) -> List[Dict[str, Union[str, bool, float]]]:
        """
        여러 이미지에 대해 일괄 추론을 수행합니다.

        Args:
            image_sources (List[str]): 이미지 URL 또는 로컬 파일 경로 리스트
            threshold (float, optional): Sigmoid threshold (None이면 초기화 시 설정한 값 사용)
            cleanup (bool): URL 임시 파일 삭제 여부

        Returns:
            List[Dict]: 각 이미지의 분류 결과 리스트
                예: [
                    {
                        "image_source": "img1.jpg",
                        "has_step": True,
                        "has_step_prob": 0.876,
                        ...
                    },
                    {
                        "image_source": "img2.jpg",
                        "has_step": False,
                        "has_step_prob": 0.234,
                        ...
                    }
                ]

        Examples:
            >>> inferencer = SigLIPInference(checkpoint_path="model.pth")
            >>> results = inferencer.batch_infer(["img1.jpg", "img2.jpg"])
            >>> print(results[0])
            {'image_source': 'img1.jpg', 'has_step': True, ...}
        """
        results = []
        temp_files = []

        try:
            for i, image_source in enumerate(image_sources):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"[{i+1}/{len(image_sources)}] 처리 중: {image_source}")
                    print(f"{'='*60}")

                is_url = image_source.startswith(('http://', 'https://'))

                # URL인 경우 이미지 다운로드
                if is_url:
                    try:
                        image_path = self.download_image_from_url(image_source)
                        temp_files.append(image_path)
                        if self.verbose:
                            print(f"이미지 다운로드 완료: {image_path}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  경고: 이미지 다운로드 실패. 건너뜁니다. ({e})")
                        continue
                else:
                    if not os.path.exists(image_source):
                        if self.verbose:
                            print(f"  경고: 파일을 찾을 수 없습니다. 건너뜁니다.")
                        continue
                    image_path = image_source

                # Multi-label classification 수행
                result = self.classifier.classify(
                    image_path=image_path,
                    threshold=threshold
                )

                # 결과 정리
                result_with_source = {
                    "image_source": image_source,
                    **result
                }

                results.append(result_with_source)

                if self.verbose:
                    print(f"\n결과:")
                    for key, value in result.items():
                        if not key.endswith('_prob'):
                            prob = result.get(f"{key}_prob", 0.0)
                            status = "✓" if value else "✗"
                            print(f"  {status} {key}: {value} (확률: {prob:.3f})")

        finally:
            # 임시 파일 정리
            if cleanup:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            if self.verbose:
                                print(f"임시 파일 삭제: {temp_file}")
                    except Exception as e:
                        if self.verbose:
                            print(f"임시 파일 삭제 실패: {e}")

        return results


if __name__ == "__main__":
    # CLI 사용 예시
    import argparse

    parser = argparse.ArgumentParser(description="SigLIP Multi-Label Classification")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="이미지 URL 또는 로컬 파일 경로"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="파인튜닝된 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold (기본값: 0.5)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Hugging Face 모델 ID"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SigLIP Multi-Label Classification")
    print("=" * 60)
    print(f"모델 ID: {args.model_id}")
    print(f"체크포인트: {args.checkpoint or '없음 (기본 모델)'}")
    print(f"임계값: {args.threshold}")
    print("=" * 60)
    print()

    # 추론 실행
    inferencer = SigLIPInference(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold
    )

    result = inferencer.infer(image_source=args.image)

    print("\n" + "=" * 60)
    print("최종 결과:")
    print("=" * 60)
    for key, value in result.items():
        if not key.endswith('_prob'):
            status = "✓" if value else "✗"
            prob = result.get(f"{key}_prob", 0.0)
            print(f"  {status} {key}: {value} (확률: {prob:.3f})")
    print("=" * 60)
