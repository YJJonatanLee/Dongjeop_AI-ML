"""
SigLIP Multi-Label Classification Training Script

Grounding DINO 학습 스크립트를 참고하여 작성되었습니다.
config 파일을 통해 모든 학습 설정을 관리합니다.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

# 프로젝트 루트를 path에 추가
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataloader.dataset import MultiLabelClassificationDataset, collate_fn, get_augment_pipeline
from models.model_loader import load_model
from training.trainer import SigLIPTrainer
from training.losses import get_loss_fn


def parse_args():
    parser = argparse.ArgumentParser(description="SigLIP Multi-Label Classification Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "models" / "configs" / "siglip_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use-mldecoder",
        action="store_true",
        help="Enable ML Decoder (overrides config ml_decoder.enabled)",
    )
    return parser.parse_args()


def resolve_data_path(relative_path: str) -> Path:
    """상대 경로를 절대 경로로 변환"""
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if candidate.exists():
        return candidate
    alt_candidate = (PROJECT_ROOT.parent / relative_path).resolve()
    if alt_candidate.exists():
        return alt_candidate
    raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {relative_path}")


def create_dataloaders(config, config_path, processor, num_workers=4):
    """데이터로더 생성"""
    # 경로 해석
    train_img_dir = resolve_data_path(config["data"]["train_img_dir"])
    train_ann_file = resolve_data_path(config["data"]["train_ann_file"])
    val_img_dir = resolve_data_path(config["data"]["val_img_dir"])
    val_ann_file = resolve_data_path(config["data"]["val_ann_file"])

    # Augmentation 파이프라인
    train_aug = get_augment_pipeline(config_path, stage="train")
    val_aug = get_augment_pipeline(config_path, stage="val")

    # 데이터셋 생성
    train_dataset = MultiLabelClassificationDataset(
        str(train_img_dir), str(train_ann_file), augment_pipeline=train_aug
    )
    val_dataset = MultiLabelClassificationDataset(
        str(val_img_dir), str(val_ann_file), augment_pipeline=val_aug
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Collate function
    def make_collate_fn():
        def _fn(batch):
            return collate_fn(batch, processor)
        return _fn

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_optimizer(model, config):
    """Optimizer 생성 (Differential Learning Rates 지원)"""
    training_cfg = config["training"]
    optimizer_cfg = training_cfg.get("optimizer", {})

    # Check if ML Decoder is enabled
    ml_decoder_enabled = config.get("ml_decoder", {}).get("enabled", False)
    use_differential_lr = ml_decoder_enabled and not training_cfg.get("freeze_vision_encoder", False)

    if use_differential_lr:
        # Differential Learning Rates for Vision Encoder and Classifier
        vision_lr = training_cfg.get("vision_encoder_lr", training_cfg["learning_rate"] * 0.1)
        classifier_lr = training_cfg.get("classifier_lr", training_cfg["learning_rate"])

        # Separate parameters
        vision_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if name.startswith("vision_model"):
                vision_params.append(param)
            else:
                classifier_params.append(param)

        param_groups = [
            {"params": vision_params, "lr": vision_lr, "name": "vision_encoder"},
            {"params": classifier_params, "lr": classifier_lr, "name": "classifier"},
        ]

        print(f"=== Differential Learning Rates ===")
        print(f"Vision Encoder LR: {vision_lr}")
        print(f"Classifier LR: {classifier_lr}")
        print(f"Vision Encoder params: {len(vision_params)}")
        print(f"Classifier params: {len(classifier_params)}")

        optimizer = AdamW(
            param_groups,
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=optimizer_cfg.get("eps", 1e-8),
            weight_decay=training_cfg.get("weight_decay", 0.01),
        )
    else:
        # Standard single learning rate
        optimizer = AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=optimizer_cfg.get("eps", 1e-8),
            weight_decay=training_cfg.get("weight_decay", 0.01),
        )

    return optimizer


def create_scheduler(optimizer, config, total_steps, steps_per_epoch):
    """Scheduler 생성 (Warmup + CosineAnnealing)"""
    import math

    training_cfg = config["training"]
    scheduler_cfg = training_cfg.get("scheduler", {})
    step_per_batch = scheduler_cfg.get("step_per_batch", True)

    # Warmup 설정
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    if step_per_batch:
        warmup_steps = warmup_epochs * steps_per_epoch
    else:
        warmup_steps = warmup_epochs

    min_lr = training_cfg.get("min_lr", 1e-7)
    base_lr = training_cfg["learning_rate"]

    print(f"=== Scheduler 설정 ===")
    print(f"warmup_epochs: {warmup_epochs}")
    print(f"warmup_steps: {warmup_steps}")
    print(f"total_steps: {total_steps}")

    # Differential LR: 각 param group마다 독립적인 lambda 함수 생성
    if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
        print(f"Param groups: {len(optimizer.param_groups)}")

        # 각 그룹의 initial_lr 대비 min_lr 비율 계산
        lr_lambdas = []
        for i, group in enumerate(optimizer.param_groups):
            initial_lr = group['lr']
            # 각 그룹마다 initial_lr에 비례하는 min_lr 계산
            group_min_lr = min_lr * (initial_lr / base_lr) if base_lr > 0 else min_lr
            min_lr_factor = group_min_lr / initial_lr

            print(f"  Group {i} ({group.get('name', 'unnamed')}): "
                  f"initial_lr={initial_lr:.2e}, min_lr={group_min_lr:.2e}, "
                  f"factor={min_lr_factor:.4f}")

            # 각 그룹용 lambda 함수 생성
            def make_lr_lambda(factor):
                def lr_lambda(current_step):
                    if warmup_steps > 0 and current_step < warmup_steps:
                        # Linear warmup: 0 -> 1
                        return current_step / warmup_steps
                    else:
                        # Cosine annealing: 1 -> factor
                        if warmup_steps > 0:
                            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                        else:
                            progress = current_step / total_steps
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                        return factor + (1 - factor) * cosine_decay
                return lr_lambda

            lr_lambdas.append(make_lr_lambda(min_lr_factor))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

        if warmup_steps > 0:
            print(f"Scheduler: Warmup ({warmup_steps} steps) + CosineAnnealing ({total_steps - warmup_steps} steps)")
            print(f"  Each group scales independently from its initial_lr to its min_lr")
        else:
            print(f"CosineAnnealingLR scheduler with T_max={total_steps}")

    # Single LR: 하나의 lambda 함수만 사용
    else:
        print(f"base_lr: {base_lr}, min_lr: {min_lr}")
        min_lr_factor = min_lr / base_lr

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                # Linear warmup: 0 -> 1
                return current_step / warmup_steps
            else:
                # Cosine annealing: 1 -> min_lr_factor
                if warmup_steps > 0:
                    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                else:
                    progress = current_step / total_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_factor + (1 - min_lr_factor) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if warmup_steps > 0:
            print(f"Scheduler: Warmup ({warmup_steps} steps) + CosineAnnealing ({total_steps - warmup_steps} steps)")
            print(f"  Warmup: 0 -> 1, Decay: 1 -> {min_lr_factor:.4f}")
        else:
            print(f"CosineAnnealingLR scheduler with T_max={total_steps}")

    return scheduler


def main():
    args = parse_args()

    # Config 로드
    print(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ML Decoder 플래그 오버라이드
    if args.use_mldecoder:
        if "ml_decoder" not in config:
            config["ml_decoder"] = {}
        config["ml_decoder"]["enabled"] = True
        print("ML Decoder enabled via --use-mldecoder flag")

    # Device 설정
    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Processor 로드
    model_id = config["model"]["pretrained_model_id"]
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

    # 데이터로더 생성
    train_loader, val_loader = create_dataloaders(
        config, args.config, processor, num_workers=args.num_workers
    )

    # 모델 로드
    model = load_model(config_path=args.config)
    model.to(device)

    # Optimizer 생성
    optimizer = create_optimizer(model, config)

    # Total steps 계산
    training_cfg = config["training"]
    steps_per_epoch = len(train_loader) // training_cfg.get("gradient_accumulation_steps", 1)
    total_steps = steps_per_epoch * training_cfg["epochs"]
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Scheduler 생성
    scheduler = create_scheduler(optimizer, config, total_steps, steps_per_epoch)

    # Threshold 설정
    threshold = config.get("metrics", {}).get("threshold", 0.5)

    # Loss 함수 생성
    loss_config = training_cfg.get("loss", {"type": "bce"})
    loss_fn = get_loss_fn(loss_config)

    # Trainer 생성
    trainer = SigLIPTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        model_name=config["model"]["name"],
        loss_fn=loss_fn,
        project_root=PROJECT_ROOT,
        threshold=threshold,
    )

    # 학습 루프
    num_epochs = training_cfg["epochs"]
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Train
        train_result = trainer.train_epoch(train_loader, epoch)

        # Validation
        val_result = trainer.evaluate_epoch(val_loader)

        # Epoch 결과 처리
        trainer.handle_epoch_end(epoch, train_result, val_result)

        # Early stopping 체크
        if trainer.should_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved at: {trainer.best_ckpt}")
    print(f"Training log saved at: {trainer.log_file}")


if __name__ == "__main__":
    main()
