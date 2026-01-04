#!/bin/bash
set -euo pipefail

# SigLIP Multi-Label Classification Inference Script
# uv가 설치되어 있으면 uv run으로 의존성을 자동 관리합니다.

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# Python 실행기 결정
PYTHON_CMD=(python)
if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
fi

# 기본 설정
IMAGE_PATH=""
CHECKPOINT_PATH=""
THRESHOLD=0.5
MODEL_ID="google/siglip-base-patch16-224"

# 도움말 함수
show_help() {
    echo "=========================================="
    echo "SigLIP Multi-Label Classification Inference"
    echo "=========================================="
    echo ""
    echo "사용법: $0 [옵션] --image <이미지_경로>"
    echo ""
    echo "필수 인자:"
    echo "  --image PATH              이미지 파일 경로"
    echo ""
    echo "선택 인자:"
    echo "  --checkpoint PATH         체크포인트 파일 경로"
    echo "  --threshold VALUE         Sigmoid 임계값 (기본값: 0.5)"
    echo "  --model-id ID            Hugging Face 모델 ID (기본값: google/siglip-base-patch16-224)"
    echo "  -h, --help               이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  # 기본 HF 모델로 추론"
    echo "  $0 --image test.jpg"
    echo ""
    echo "  # 파인튜닝된 모델로 추론"
    echo "  $0 --image test.jpg --checkpoint models/checkpoints/siglip-base-patch16-224/best_model.pth"
    echo ""
    echo "  # 임계값 조정"
    echo "  $0 --image test.jpg --checkpoint models/checkpoints/siglip-base-patch16-224/best_model.pth --threshold 0.6"
    echo ""
    echo "  # SigLIP2 사용"
    echo "  $0 --image test.jpg --model-id google/siglip2-base-patch16-224"
    echo "=========================================="
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말을 보려면 '$0 --help'를 실행하세요."
            exit 1
            ;;
    esac
done

# 이미지 경로 확인
if [ -z "$IMAGE_PATH" ]; then
    echo "오류: 이미지 경로를 지정해주세요."
    echo "도움말을 보려면 '$0 --help'를 실행하세요."
    exit 1
fi

# Python 명령어 구성
CMD=("${PYTHON_CMD[@]}" infer/classifier.py --image "$IMAGE_PATH" --threshold "$THRESHOLD" --model-id "$MODEL_ID")

if [ -n "$CHECKPOINT_PATH" ]; then
    CMD+=("--checkpoint" "$CHECKPOINT_PATH")
fi

echo "=========================================="
echo "SigLIP Multi-Label Classification Inference"
echo "=========================================="
echo "이미지: $IMAGE_PATH"
echo "모델 ID: $MODEL_ID"
echo "임계값: $THRESHOLD"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "체크포인트: $CHECKPOINT_PATH"
else
    echo "체크포인트: 없음 (HF 기본 모델)"
fi
echo "=========================================="
echo ""

# 추론 실행
"${CMD[@]}"

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
