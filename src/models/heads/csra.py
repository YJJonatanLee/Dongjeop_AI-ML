"""
CSRA: Class-Specific Residual Attention

Based on the paper: "Residual Attention: A Simple but Effective Method for Multi-Label Recognition"
ICCV 2021
https://arxiv.org/abs/2108.02456

Official implementation: https://github.com/Kevinz-code/CSRA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSRAHead(nn.Module):
    """
    Single CSRA head with a specific temperature.

    Applies class-specific attention over spatial features with residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        temperature: float = 1.0,
        lam: float = 0.1,
    ):
        """
        Args:
            input_dim (int): Input feature dimension from vision encoder
            num_classes (int): Number of output classes
            temperature (float): Temperature for softmax attention (higher = softer)
            lam (float): Weight for residual connection (0~1)
        """
        super().__init__()
        self.temperature = temperature
        self.lam = lam
        self.num_classes = num_classes

        # Class-specific projection
        self.head = nn.Conv1d(input_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Spatial features from vision encoder
                       Shape: [batch, seq_len, input_dim]

        Returns:
            Tensor: Class logits, shape [batch, num_classes]
        """
        # x: [B, N, C] -> [B, C, N] for Conv1d
        x = x.transpose(1, 2)

        # score: [B, num_classes, N]
        score = self.head(x) / self.temperature

        # base: average pooled logits (GAP)
        # [B, num_classes, N] -> [B, num_classes]
        base = score.mean(dim=-1)

        # Class-specific attention: softmax over spatial dimension
        # [B, num_classes, N]
        attn = F.softmax(score, dim=-1)

        # Attention-weighted sum of scores
        # [B, num_classes]
        score_attn = (attn * score).sum(dim=-1)

        # Residual combination
        # lam controls the balance between attention and average pooling
        logits = self.lam * score_attn + (1 - self.lam) * base

        return logits


class CSRA(nn.Module):
    """
    Multi-Head Class-Specific Residual Attention.

    Uses multiple CSRA heads with different temperatures to capture
    features at different granularity levels.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 4,
        lam: float = 0.1,
        temperatures: list = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim (int): Input feature dimension from vision encoder
            num_classes (int): Number of output classes
            num_heads (int): Number of CSRA heads with different temperatures
            lam (float): Weight for residual connection
            temperatures (list, optional): Custom temperature list for each head
                                          If None, uses [1, 2, 4, 6] pattern
            dropout (float): Dropout rate before CSRA heads
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads

        # Default temperatures if not provided
        if temperatures is None:
            # Common pattern: [1, 2, 4, 6] for 4 heads
            temperatures = [1.0 * (2 ** i) if i < 2 else 2.0 * (i + 1)
                          for i in range(num_heads)]
            # Simplified: [1.0, 2.0, 4.0, 6.0]
            temperatures = [1.0, 2.0, 4.0, 6.0][:num_heads]

        self.temperatures = temperatures

        # Dropout before heads
        self.dropout = nn.Dropout(dropout)

        # Multiple CSRA heads with different temperatures
        self.heads = nn.ModuleList([
            CSRAHead(
                input_dim=input_dim,
                num_classes=num_classes,
                temperature=t,
                lam=lam,
            )
            for t in temperatures
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for head in self.heads:
            nn.init.xavier_uniform_(head.head.weight)
            if head.head.bias is not None:
                nn.init.zeros_(head.head.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Spatial features from vision encoder
                       Shape: [batch, seq_len, input_dim]
                       or [batch, input_dim] (will be unsqueezed)

        Returns:
            Tensor: Class logits, shape [batch, num_classes]
        """
        # Handle both pooled and sequence features
        if x.dim() == 2:
            # Pooled features: [batch, input_dim] -> [batch, 1, input_dim]
            x = x.unsqueeze(1)

        # Apply dropout
        x = self.dropout(x)

        # Apply all heads and average
        outputs = [head(x) for head in self.heads]
        logits = sum(outputs) / len(outputs)

        return logits


if __name__ == "__main__":
    # Test CSRA
    print("=== Testing CSRA ===")

    batch_size = 4
    num_classes = 5
    feature_dim = 768  # Vision encoder hidden size
    seq_len = 196  # 14x14 patches

    # Create CSRA
    csra = CSRA(
        input_dim=feature_dim,
        num_classes=num_classes,
        num_heads=4,
        lam=0.1,
        dropout=0.1,
    )

    print(f"CSRA created with {num_classes} classes")
    print(f"Temperatures: {csra.temperatures}")
    print(f"Total parameters: {sum(p.numel() for p in csra.parameters()):,}")

    # Test with sequence features
    seq_features = torch.randn(batch_size, seq_len, feature_dim)
    logits = csra(seq_features)
    print(f"\nSequence features input: {seq_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    # Test with pooled features
    pooled_features = torch.randn(batch_size, feature_dim)
    logits = csra(pooled_features)
    print(f"\nPooled features input: {pooled_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    print("\n CSRA test passed!")
