"""
Custom Loss Functions for Multi-Label Classification

Includes:
- Focal Loss: Focus on hard examples
- ASL (Asymmetric Loss): Asymmetric treatment for positive/negative samples
- BCE Loss: Standard binary cross-entropy (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha (float): Weighting factor in [0, 1] for class 1 (default: 0.25)
        gamma (float): Focusing parameter γ >= 0 (default: 2.0)
        reduction (str): 'mean' or 'sum' or 'none'

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] - raw logits
            targets: [batch_size, num_classes] - binary labels (0 or 1)

        Returns:
            loss: scalar tensor
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Get probabilities
        probs = torch.sigmoid(logits)

        # p_t: probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification

    Treats positive and negative samples asymmetrically.
    Better for imbalanced multi-label tasks.

    Args:
        gamma_neg (float): Focusing parameter for negative samples (default: 4)
        gamma_pos (float): Focusing parameter for positive samples (default: 1)
        clip (float): Probability margin (default: 0.05)
        eps (float): Numerical stability (default: 1e-8)
        reduction (str): 'mean' or 'sum' or 'none'

    Reference:
        Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] - raw logits
            targets: [batch_size, num_classes] - binary labels (0 or 1)

        Returns:
            loss: scalar tensor
        """
        # Probabilities
        probs = torch.sigmoid(logits)

        # Probability margin (prevent overconfidence)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Positive loss (for labels = 1)
        # Focus on hard positives (gamma_pos = 1, less focusing)
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)

        # Negative loss (for labels = 0)
        # Focus on hard negatives (gamma_neg = 4, more focusing)
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss_neg = loss_neg * (probs_pos ** self.gamma_neg)

        # Total loss
        loss = -loss_pos - loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCEWithLogitsLoss(nn.Module):
    """
    Standard Binary Cross-Entropy with Logits Loss
    (Wrapper for consistency)
    """

    def __init__(self, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


def get_loss_fn(loss_config):
    """
    Loss 함수 팩토리

    Args:
        loss_config (dict): Loss 설정
            {
                'type': 'focal' | 'asl' | 'bce',
                'params': {...}
            }

    Returns:
        loss_fn: Loss 함수
    """
    loss_type = loss_config.get('type', 'bce').lower()
    loss_params = loss_config.get('params', {})

    if loss_type == 'focal':
        print("Using Focal Loss")
        print(f"  - alpha: {loss_params.get('alpha', 0.25)}")
        print(f"  - gamma: {loss_params.get('gamma', 2.0)}")
        return FocalLoss(
            alpha=loss_params.get('alpha', 0.25),
            gamma=loss_params.get('gamma', 2.0),
            reduction='mean'
        )

    elif loss_type == 'asl':
        print("Using Asymmetric Loss (ASL)")
        print(f"  - gamma_neg: {loss_params.get('gamma_neg', 4)}")
        print(f"  - gamma_pos: {loss_params.get('gamma_pos', 1)}")
        print(f"  - clip: {loss_params.get('clip', 0.05)}")
        return AsymmetricLoss(
            gamma_neg=loss_params.get('gamma_neg', 4),
            gamma_pos=loss_params.get('gamma_pos', 1),
            clip=loss_params.get('clip', 0.05),
            reduction='mean'
        )

    elif loss_type == 'bce':
        print("Using BCE with Logits Loss")
        return BCEWithLogitsLoss(reduction='mean')

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from ['focal', 'asl', 'bce']")


if __name__ == '__main__':
    # 테스트 코드
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # 샘플 데이터
    batch_size = 4
    num_classes = 5

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    print(f"\nLogits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"\nTargets:\n{targets}")

    # BCE Loss
    print("\n" + "-" * 60)
    bce_loss_fn = BCEWithLogitsLoss()
    bce_loss = bce_loss_fn(logits, targets)
    print(f"BCE Loss: {bce_loss.item():.4f}")

    # Focal Loss
    print("\n" + "-" * 60)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(logits, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")

    # ASL
    print("\n" + "-" * 60)
    asl_loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    asl_loss = asl_loss_fn(logits, targets)
    print(f"ASL: {asl_loss.item():.4f}")

    # Config 테스트
    print("\n" + "=" * 60)
    print("Testing Loss Factory")
    print("=" * 60)

    configs = [
        {'type': 'bce', 'params': {}},
        {'type': 'focal', 'params': {'alpha': 0.25, 'gamma': 2.0}},
        {'type': 'asl', 'params': {'gamma_neg': 4, 'gamma_pos': 1, 'clip': 0.05}}
    ]

    for config in configs:
        print(f"\nConfig: {config}")
        loss_fn = get_loss_fn(config)
        loss = loss_fn(logits, targets)
        print(f"Loss: {loss.item():.4f}")
