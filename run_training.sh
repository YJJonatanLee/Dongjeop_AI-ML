#!/bin/bash
set -euo pipefail

# SigLIP Multi-Label Classification Training Script
# uv가 설치되어 있으면 uv run으로 의존성을 자동 관리합니다.

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# Python 실행기 결정
PYTHON_CMD=(python)
if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
fi

# 도움말 함수
show_help() {
    echo "=========================================="
    echo "SigLIP Multi-Label Classification Training"
    echo "=========================================="
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  --config PATH         커스텀 config 파일 경로"
    echo "  --model-type TYPE     모델 타입 선택 (siglip, siglip-mldecoder, siglip2, siglip2-mldecoder)"
    echo "  --num-workers N       데이터로더 워커 수 (기본값: 4)"
    echo "  -h, --help           이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  # SigLIP 학습 (linear head)"
    echo "  $0 --model-type siglip"
    echo ""
    echo "  # SigLIP + ML Decoder 학습 (권장!)"
    echo "  $0 --model-type siglip-mldecoder"
    echo ""
    echo "  # SigLIP2 학습 (linear head)"
    echo "  $0 --model-type siglip2"
    echo ""
    echo "  # SigLIP2 + ML Decoder 학습 (최고 성능!)"
    echo "  $0 --model-type siglip2-mldecoder"
    echo ""
    echo "  # 커스텀 config 파일 사용"
    echo "  $0 --config models/configs/my_config.yaml"
    echo ""
    echo "  # 워커 수 지정"
    echo "  $0 --model-type siglip-mldecoder --num-workers 8"
    echo "=========================================="
}

# 기본 설정
CONFIG_PATH=""
MODEL_TYPE=""
NUM_WORKERS=4

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
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

# Config 경로 및 ML Decoder 플래그 결정
USE_MLDECODER=""
if [ -z "$CONFIG_PATH" ]; then
    if [ "$MODEL_TYPE" = "siglip" ]; then
        CONFIG_PATH="models/configs/siglip_config.yaml"
    elif [ "$MODEL_TYPE" = "siglip-mldecoder" ]; then
        CONFIG_PATH="models/configs/siglip_config.yaml"
        USE_MLDECODER="--use-mldecoder"
    elif [ "$MODEL_TYPE" = "siglip2" ]; then
        CONFIG_PATH="models/configs/siglip2_config.yaml"
    elif [ "$MODEL_TYPE" = "siglip2-mldecoder" ]; then
        CONFIG_PATH="models/configs/siglip2_config.yaml"
        USE_MLDECODER="--use-mldecoder"
    else
        # 기본값: siglip
        CONFIG_PATH="models/configs/siglip_config.yaml"
        MODEL_TYPE="siglip"
    fi
fi

# Config 파일 존재 확인
if [ ! -f "$CONFIG_PATH" ]; then
    echo "오류: Config 파일을 찾을 수 없습니다: $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "SigLIP Multi-Label Classification Training"
echo "=========================================="
echo "모델 타입: $MODEL_TYPE"
echo "Config: $CONFIG_PATH"
echo "워커 수: $NUM_WORKERS"
echo "=========================================="
echo ""

# 학습 실행
"${PYTHON_CMD[@]}" src/training/train.py --config "$CONFIG_PATH" --num_workers "$NUM_WORKERS" $USE_MLDECODER

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
