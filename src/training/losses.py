"""
Custom Loss Functions for Multi-Label Classification

Includes:
- ASL (Asymmetric Loss): Asymmetric treatment for positive/negative samples (recommended)
- BCE Loss: Standard binary cross-entropy (baseline)
"""

import torch
import torch.nn as nn


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
                'type': 'asl' | 'bce',
                'params': {...}
            }

    Returns:
        loss_fn: Loss 함수
    """
    loss_type = loss_config.get('type', 'bce').lower()
    loss_params = loss_config.get('params', {})

    if loss_type == 'asl':
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
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from ['asl', 'bce']")


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
        {'type': 'asl', 'params': {'gamma_neg': 4, 'gamma_pos': 1, 'clip': 0.05}}
    ]

    for config in configs:
        print(f"\nConfig: {config}")
        loss_fn = get_loss_fn(config)
        loss = loss_fn(logits, targets)
        print(f"Loss: {loss.item():.4f}")
