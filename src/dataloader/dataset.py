import os
import json
import yaml
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Dict


# 레이블 순서 정의 (width 제외)
LABEL_NAMES = [
    "has_step",
    "has_movable_chair",
    "has_high_chair",
    "has_fixed_chair",
    "has_floor_chair"
]


def get_augment_pipeline(config_path, stage='train'):
    """
    YAML 설정 파일에서 증강 설정을 읽어 albumentations 파이프라인을 생성합니다.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    aug_config = config.get('augmentations', {})
    pipeline = []

    if aug_config.get('enabled', False):
        stage_transforms = aug_config.get(stage, [])
        for aug in stage_transforms:
            name = aug['name']
            if name == "ToTensorV2":
                print("경고: ToTensorV2는 processor에서 처리하므로 건너뜁니다.")
                continue
            params = aug.get('params', {})
            if hasattr(A, name):
                pipeline.append(getattr(A, name)(**params))
            else:
                print(f"경고: albumentations에 '{name}' 증강이 없습니다. 건너뜁니다.")

    return A.Compose(pipeline)


class MultiLabelClassificationDataset(Dataset):
    """
    Multi-label classification을 위한 데이터셋 클래스

    각 이미지는 5개의 binary label을 가짐:
    - has_step
    - has_movable_chair
    - has_high_chair
    - has_fixed_chair
    - has_floor_chair
    """
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        augment_pipeline: Optional[A.Compose] = None,
        label_names: Optional[List[str]] = None
    ):
        self.image_dir = image_dir
        self.augment_pipeline = augment_pipeline
        self.label_names = label_names or LABEL_NAMES

        # 어노테이션 로드
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        print(f"데이터셋 로드 완료: {len(self.annotations)}개 샘플")

        # 클래스 분포 확인
        self._print_class_distribution()

    def _print_class_distribution(self):
        """클래스 분포 출력"""
        counts = {label: 0 for label in self.label_names}

        for ann in self.annotations:
            for label in self.label_names:
                if ann.get(label, False):
                    counts[label] += 1

        print("\n클래스 분포:")
        for label, count in counts.items():
            ratio = count / len(self.annotations) * 100
            print(f"  {label}: {count} ({ratio:.1f}%)")
        print()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # 이미지 로드
        image_path = os.path.join(self.image_dir, ann['image'])

        # PIL Image로 로드 (RGB)
        image = Image.open(image_path).convert("RGB")

        # Augmentation (albumentations는 numpy array 사용)
        if self.augment_pipeline:
            image_np = np.array(image)
            augmented = self.augment_pipeline(image=image_np)
            image = Image.fromarray(augmented['image'])

        # Multi-label 생성 (float tensor for BCE loss)
        labels = torch.tensor([
            float(ann.get(label, False))
            for label in self.label_names
        ], dtype=torch.float32)

        return {
            "image": image,
            "labels": labels,
            "image_id": ann.get('image_id', ann['image'])
        }


def collate_fn(batch, processor):
    """
    배치 단위로 데이터를 처리하기 위한 콜레이트 함수.
    SigLIP processor가 이미지 전처리를 수행합니다.
    """
    images = [item["image"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # Processor로 이미지 전처리
    # SigLIP은 모든 이미지를 고정 크기(224x224)로 리사이즈하므로 padding 불필요
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    # labels 추가
    inputs["labels"] = labels

    return inputs


if __name__ == '__main__':
    from transformers import AutoImageProcessor
    from torch.utils.data import DataLoader

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'models/configs/siglip_config.yaml')

    # Config 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    train_img_dir = os.path.join(project_root, '..', data_config['train_img_dir'])
    train_ann_file = os.path.join(project_root, '..', data_config['train_ann_file'])

    # 1. 증강 파이프라인 생성
    augment_pipeline = get_augment_pipeline(config_path, stage='train')
    print("--- 생성된 증강 파이프라인 ---")
    print(augment_pipeline)

    # 2. 데이터셋 생성
    dataset = MultiLabelClassificationDataset(
        image_dir=train_img_dir,
        annotation_file=train_ann_file,
        augment_pipeline=augment_pipeline
    )
    print(f"\n데이터셋 크기: {len(dataset)}")
    print(f"레이블: {dataset.label_names}")

    # 3. 프로세서 로드
    model_id = config['model']['pretrained_model_id']
    processor = AutoImageProcessor.from_pretrained(model_id)

    # 4. DataLoader 테스트
    def dynamic_collate_fn(batch):
        return collate_fn(batch, processor)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dynamic_collate_fn)

    # 첫 번째 배치 확인
    try:
        batch_data = next(iter(dataloader))
        print("\n--- 데이터로더 배치 샘플 ---")
        print("입력 키:", batch_data.keys())
        print("pixel_values shape:", batch_data['pixel_values'].shape)
        print("labels shape:", batch_data['labels'].shape)
        print("labels 샘플 (첫 번째 이미지):", batch_data['labels'][0])
        print("  → has_step:", batch_data['labels'][0][0].item())
        print("  → has_movable_chair:", batch_data['labels'][0][1].item())

    except Exception as e:
        print(f"\n데이터로더 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
