import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from collections.abc import Mapping

import torch
from torch import nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score

def _move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors to the target device."""
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, Mapping):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(item, device) for item in data)
    return data


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]


class SigLIPTrainer:
    """
    Multi-label classification을 위한 SigLIP Trainer.
    Grounding DINO Trainer 구조를 유지하되, task에 맞게 수정.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        model_name: str,
        loss_fn: Optional[nn.Module] = None,
        project_root: Optional[Path] = None,
        threshold: float = 0.5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.model_name = model_name
        self.project_root = project_root or Path.cwd()
        self.threshold = threshold
        self.loss_fn = loss_fn  # Custom loss function

        training_cfg = config.get("training", {})
        self.grad_accum_steps = max(1, training_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
        self.use_amp = device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = GradScaler('cpu', enabled=False)

        # Best metric configuration
        self.best_metric = training_cfg.get("best_metric", "loss")
        # Metrics where higher is better
        self.higher_is_better = self.best_metric != "loss"

        # Early stopping
        self.early_stopping_patience = training_cfg.get("early_stopping_patience", 0)
        self.early_stopping_counter = 0
        self.should_stop = False

        output_cfg = config.get("output", {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_timestamp = timestamp

        log_root = self.project_root / output_cfg.get("log_dir", "outputs/logs")
        ckpt_root = self.project_root / output_cfg.get("checkpoint_dir", "models/checkpoints")
        self.log_dir = log_root / model_name
        self.checkpoint_dir = ckpt_root / model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"train_{timestamp}.log"
        self.best_ckpt = self.checkpoint_dir / f"best_model_{model_name}_{timestamp}.pth"

        # Initialize best metric value based on optimization direction
        if self.higher_is_better:
            self.best_metric_value = float("-inf")
        else:
            self.best_metric_value = float("inf")

        self._log(f"Training initialized for {model_name} at {timestamp}")
        self._log(f"Best model selection metric: {self.best_metric} ({'maximize' if self.higher_is_better else 'minimize'})")

    def train_epoch(self, dataloader: Iterable[Dict[str, Any]], epoch: int) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        pending_step = False
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        progress = tqdm(
            enumerate(dataloader),
            total=total_batches,
            desc=f"Train Epoch {epoch + 1}",
            leave=False,
        )
        for step, batch in progress:
            batch = _move_to_device(batch, self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # Custom loss 사용
                if self.loss_fn is not None:
                    # Custom loss: logits만 필요
                    outputs = self.model(pixel_values=batch["pixel_values"])
                    logits = outputs.logits
                    labels = batch["labels"]
                    loss = self.loss_fn(logits, labels)
                else:
                    # 기본 loss (모델 내장)
                    outputs = self.model(**batch)
                    loss = outputs.loss

            loss_value = loss.item()
            total_loss += loss_value
            num_steps += 1
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{current_lr:.2e}")

            loss = loss / self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            pending_step = True
            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending_step = False

        # 남은 gradient 처리
        if pending_step:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, num_steps)
        return EpochResult(loss=avg_loss, metrics={})

    def evaluate_epoch(self, dataloader: Iterable[Dict[str, Any]]) -> EpochResult:
        self.model.eval()
        total_loss = 0.0
        num_steps = 0

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
            progress = tqdm(
                dataloader,
                total=total_batches,
                desc="Validation",
                leave=False,
            )
            for batch in progress:
                batch = _move_to_device(batch, self.device)

                # Custom loss 사용
                if self.loss_fn is not None:
                    outputs = self.model(pixel_values=batch["pixel_values"])
                    logits = outputs.logits
                    labels = batch["labels"]
                    loss = self.loss_fn(logits, labels)
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                total_loss += loss.item()
                num_steps += 1

                # Multi-label prediction
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).int()

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

                progress.set_postfix(loss=f"{loss.item():.4f}")

        if num_steps == 0:
            return EpochResult(loss=0.0, metrics={})

        avg_loss = total_loss / num_steps

        # 메트릭 계산
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        metrics = self._compute_metrics(all_preds, all_probs, all_labels)

        return EpochResult(loss=avg_loss, metrics=metrics)

    def _compute_metrics(self, preds: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Multi-label classification 메트릭 계산"""
        # Subset accuracy (전체 레이블이 정확히 일치)
        subset_acc = accuracy_score(labels, preds)

        # Per-label metrics (micro/macro average)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )

        # Hamming accuracy (per-label accuracy)
        hamming_acc = (preds == labels).mean()

        # mAP (mean Average Precision) - requires probabilities
        try:
            mAP_macro = average_precision_score(labels, probs, average='macro')
            mAP_micro = average_precision_score(labels, probs, average='micro')
        except ValueError:
            # Handle edge case where all labels are 0 or 1
            mAP_macro = 0.0
            mAP_micro = 0.0

        return {
            "subset_accuracy": subset_acc,
            "hamming_accuracy": hamming_acc,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "mAP_macro": mAP_macro,
            "mAP_micro": mAP_micro,
        }

    def handle_epoch_end(
        self,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult,
    ) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Get current metric value
        if self.best_metric == "loss":
            current_metric_value = val_result.loss
        else:
            current_metric_value = val_result.metrics.get(self.best_metric, float("-inf") if self.higher_is_better else float("inf"))

        # Check if this is the best model
        if self.higher_is_better:
            is_best = current_metric_value > self.best_metric_value
        else:
            is_best = current_metric_value < self.best_metric_value

        if is_best:
            self.best_metric_value = current_metric_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        # 로그 출력
        block = "=" * 60
        self._log(block)
        self._log(f"Epoch {epoch + 1}")
        self._log(f"Train Loss: {train_result.loss:.4f}")
        self._log(f"Validation Loss: {val_result.loss:.4f}")

        # 메트릭 출력
        if val_result.metrics:
            self._log("Validation Metrics:")
            for key, value in val_result.metrics.items():
                self._log(f"  {key}: {value:.4f}")

        # Best metric 출력
        best_metric_str = "N/A" if math.isinf(self.best_metric_value) else f"{self.best_metric_value:.4f}"
        self._log(f"Best {self.best_metric}: {best_metric_str}")
        self._log(f"Learning Rate: {current_lr:.6f}")

        self._log(block)

        # Best model 저장
        if is_best:
            self._save_best_checkpoint(epoch, val_result.loss, val_result.metrics)
        else:
            if self.early_stopping_patience > 0:
                self._log(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.should_stop = True
                    self._log(f"Early stopping triggered.")

    def _save_best_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]) -> None:
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "metrics": metrics,
            "best_metric": self.best_metric,
            "best_metric_value": self.best_metric_value,
            "config": self.config,
            "model_name": self.model_name,
            "timestamp": self.run_timestamp,
        }

        torch.save(state, self.best_ckpt)
        self._log(f"Best model checkpoint saved: {self.best_ckpt} ({self.best_metric}={self.best_metric_value:.4f})")

    def _log(self, message: str) -> None:
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + os.linesep)
