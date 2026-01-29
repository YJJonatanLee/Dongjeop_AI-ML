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
    echo "Multi-Label Classification Training"
    echo "=========================================="
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  --config PATH         커스텀 config 파일 경로"
    echo "  --model-type TYPE     모델 타입 선택"
    echo "  --num-workers N       데이터로더 워커 수 (기본값: 4)"
    echo "  -h, --help           이 도움말 표시"
    echo ""
    echo "사용 가능한 모델 타입:"
    echo "  SigLIP:"
    echo "    siglip              SigLIP (linear head)"
    echo "    siglip-mldecoder    SigLIP + ML Decoder"
    echo "    siglip-csra         SigLIP + CSRA"
    echo "    siglip-q2l          SigLIP + Q2L (Query2Label)"
    echo "    siglip2             SigLIP2 (linear head)"
    echo "    siglip2-mldecoder   SigLIP2 + ML Decoder"
    echo "    siglip2-csra        SigLIP2 + CSRA"
    echo "    siglip2-q2l         SigLIP2 + Q2L (Query2Label)"
    echo ""
    echo "  DINOv2:"
    echo "    dinov2              DINOv2 (linear head)"
    echo "    dinov2-mldecoder    DINOv2 + ML Decoder (추천!)"
    echo "    dinov2-csra         DINOv2 + CSRA"
    echo "    dinov2-q2l          DINOv2 + Q2L (Query2Label)"
    echo ""
    echo "  ConvNeXt:"
    echo "    convnext            ConvNeXt (linear head)"
    echo "    convnext-mldecoder  ConvNeXt + ML Decoder"
    echo "    convnext-csra       ConvNeXt + CSRA"
    echo "    convnext-q2l        ConvNeXt + Q2L (Query2Label)"
    echo ""
    echo "  ConvNeXt V2:"
    echo "    convnextv2            ConvNeXt V2 (linear head)"
    echo "    convnextv2-mldecoder  ConvNeXt V2 + ML Decoder"
    echo "    convnextv2-csra       ConvNeXt V2 + CSRA"
    echo "    convnextv2-q2l        ConvNeXt V2 + Q2L (Query2Label)"
    echo ""
    echo "  SAM-2 (Hiera Encoder):"
    echo "    sam2                  SAM-2 (linear head)"
    echo "    sam2-mldecoder        SAM-2 + ML Decoder"
    echo "    sam2-csra             SAM-2 + CSRA"
    echo "    sam2-q2l              SAM-2 + Q2L (Query2Label)"
    echo ""
    echo "  EVA-02:"
    echo "    eva02                 EVA-02 (linear head)"
    echo "    eva02-mldecoder       EVA-02 + ML Decoder"
    echo "    eva02-csra            EVA-02 + CSRA"
    echo "    eva02-q2l             EVA-02 + Q2L (Query2Label)"
    echo ""
    echo "예시:"
    echo "  # DINOv2 + ML Decoder 학습 (추천!)"
    echo "  $0 --model-type dinov2-mldecoder"
    echo ""
    echo "  # ConvNeXt 학습"
    echo "  $0 --model-type convnext-mldecoder"
    echo ""
    echo "  # SigLIP2 + ML Decoder 학습"
    echo "  $0 --model-type siglip2-mldecoder"
    echo ""
    echo "  # 커스텀 config 파일 사용"
    echo "  $0 --config models/configs/my_config.yaml"
    echo ""
    echo "  # 워커 수 지정"
    echo "  $0 --model-type dinov2-mldecoder --num-workers 8"
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

# Config 경로 및 Head 플래그 결정
USE_MLDECODER=""
USE_CSRA=""
USE_Q2L=""
if [ -z "$CONFIG_PATH" ]; then
    case "$MODEL_TYPE" in
        siglip)
            CONFIG_PATH="models/configs/siglip_config.yaml"
            ;;
        siglip-mldecoder)
            CONFIG_PATH="models/configs/siglip_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        siglip-csra)
            CONFIG_PATH="models/configs/siglip_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        siglip-q2l)
            CONFIG_PATH="models/configs/siglip_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        siglip2)
            CONFIG_PATH="models/configs/siglip2_config.yaml"
            ;;
        siglip2-mldecoder)
            CONFIG_PATH="models/configs/siglip2_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        siglip2-csra)
            CONFIG_PATH="models/configs/siglip2_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        siglip2-q2l)
            CONFIG_PATH="models/configs/siglip2_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        dinov2)
            CONFIG_PATH="models/configs/dinov2_config.yaml"
            ;;
        dinov2-mldecoder)
            CONFIG_PATH="models/configs/dinov2_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        dinov2-csra)
            CONFIG_PATH="models/configs/dinov2_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        dinov2-q2l)
            CONFIG_PATH="models/configs/dinov2_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        convnext)
            CONFIG_PATH="models/configs/convnext_config.yaml"
            ;;
        convnext-mldecoder)
            CONFIG_PATH="models/configs/convnext_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        convnext-csra)
            CONFIG_PATH="models/configs/convnext_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        convnext-q2l)
            CONFIG_PATH="models/configs/convnext_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        convnextv2)
            CONFIG_PATH="models/configs/convnextv2_config.yaml"
            ;;
        convnextv2-mldecoder)
            CONFIG_PATH="models/configs/convnextv2_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        convnextv2-csra)
            CONFIG_PATH="models/configs/convnextv2_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        convnextv2-q2l)
            CONFIG_PATH="models/configs/convnextv2_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        sam2)
            CONFIG_PATH="models/configs/sam2_config.yaml"
            ;;
        sam2-mldecoder)
            CONFIG_PATH="models/configs/sam2_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        sam2-csra)
            CONFIG_PATH="models/configs/sam2_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        sam2-q2l)
            CONFIG_PATH="models/configs/sam2_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        eva02)
            CONFIG_PATH="models/configs/eva02_config.yaml"
            ;;
        eva02-mldecoder)
            CONFIG_PATH="models/configs/eva02_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            ;;
        eva02-csra)
            CONFIG_PATH="models/configs/eva02_config.yaml"
            USE_CSRA="--use-csra"
            ;;
        eva02-q2l)
            CONFIG_PATH="models/configs/eva02_config.yaml"
            USE_Q2L="--use-q2l"
            ;;
        *)
            # 기본값: dinov2-mldecoder (추천 모델)
            CONFIG_PATH="models/configs/dinov2_config.yaml"
            USE_MLDECODER="--use-mldecoder"
            MODEL_TYPE="dinov2-mldecoder"
            ;;
    esac
fi

# Config 파일 존재 확인
if [ ! -f "$CONFIG_PATH" ]; then
    echo "오류: Config 파일을 찾을 수 없습니다: $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "Multi-Label Classification Training"
echo "=========================================="
echo "모델 타입: $MODEL_TYPE"
echo "Config: $CONFIG_PATH"
echo "워커 수: $NUM_WORKERS"
echo "=========================================="
echo ""

# 학습 실행
"${PYTHON_CMD[@]}" src/training/train.py --config "$CONFIG_PATH" --num_workers "$NUM_WORKERS" $USE_MLDECODER $USE_CSRA $USE_Q2L

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
