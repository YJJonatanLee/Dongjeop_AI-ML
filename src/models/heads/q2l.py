"""
Q2L: Query2Label - A Simple Transformer Way to Multi-Label Classification

Based on the paper: "Query2Label: A Simple Transformer Way to Multi-Label Classification"
https://arxiv.org/abs/2107.10834

Official implementation: https://github.com/SlongLiu/query2labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Q2LDecoderLayer(nn.Module):
    """
    Single Q2L Decoder Layer with self-attention and cross-attention.

    Similar to standard Transformer decoder but optimized for multi-label classification.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        # Self-attention for label embeddings (models label correlations)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Cross-attention: labels attend to image features
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, label_embed, image_features):
        """
        Args:
            label_embed: Label embeddings [batch, num_labels, d_model]
            image_features: Image features [batch, num_patches, d_model]

        Returns:
            Updated label embeddings [batch, num_labels, d_model]
        """
        # Self-attention on label embeddings (models label correlations)
        label_embed2, _ = self.self_attn(label_embed, label_embed, label_embed)
        label_embed = label_embed + self.dropout1(label_embed2)
        label_embed = self.norm1(label_embed)

        # Cross-attention: labels attend to image features
        label_embed2, _ = self.cross_attn(label_embed, image_features, image_features)
        label_embed = label_embed + self.dropout2(label_embed2)
        label_embed = self.norm2(label_embed)

        # Feedforward
        label_embed2 = self.ffn(label_embed)
        label_embed = label_embed + self.dropout3(label_embed2)
        label_embed = self.norm3(label_embed)

        return label_embed


class Q2L(nn.Module):
    """
    Query2Label: Transformer-based Multi-Label Classification Head.

    Uses learnable label embeddings as queries that attend to image features
    through cross-attention, while modeling label correlations through self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        """
        Args:
            input_dim (int): Input feature dimension from vision encoder
            num_classes (int): Number of output classes (labels)
            num_layers (int): Number of decoder layers
            num_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
            use_pos_encoding (bool): Whether to use positional encoding for image features
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_pos_encoding = use_pos_encoding

        # Learnable label embeddings (one per class)
        self.label_embed = nn.Embedding(num_classes, input_dim)

        # Optional: Project input features if needed
        self.input_proj = nn.Identity()

        # Positional encoding for image features (optional but recommended)
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(input_dim)
        else:
            self.pos_encoding = None

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            Q2LDecoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final classifier (one output per label)
        self.classifier = nn.Linear(input_dim, 1)

        # Dropout before decoder
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.label_embed.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Image features from vision encoder
                       Shape: [batch, num_patches, input_dim]
                       or [batch, input_dim] (will be unsqueezed)

        Returns:
            Tensor: Class logits, shape [batch, num_classes]
        """
        # Handle pooled features
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        batch_size = x.size(0)

        # Project input features
        image_features = self.input_proj(x)  # [batch, num_patches, input_dim]

        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            image_features = self.pos_encoding(image_features)

        # Apply dropout
        image_features = self.dropout(image_features)

        # Get label embeddings and expand for batch
        # [num_classes, input_dim] -> [batch, num_classes, input_dim]
        label_embed = self.label_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            label_embed = layer(label_embed, image_features)

        # Classify each label embedding
        # [batch, num_classes, input_dim] -> [batch, num_classes, 1] -> [batch, num_classes]
        logits = self.classifier(label_embed).squeeze(-1)

        return logits


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for image features.

    Adds learnable or sinusoidal positional embeddings to image patch features.
    """

    def __init__(self, d_model: int, max_len: int = 1024, learnable: bool = True):
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable

        if learnable:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Sinusoidal positional encoding
            pos_embed = self._sinusoidal_encoding(max_len, d_model)
            self.register_buffer('pos_embed', pos_embed)

    def _sinusoidal_encoding(self, max_len: int, d_model: int):
        """Generate sinusoidal positional encoding"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_embed = torch.zeros(1, max_len, d_model)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]


if __name__ == "__main__":
    # Test Q2L
    print("=== Testing Q2L ===")

    batch_size = 4
    num_classes = 5
    feature_dim = 768  # Vision encoder hidden size
    seq_len = 196  # 14x14 patches

    # Create Q2L
    q2l = Q2L(
        input_dim=feature_dim,
        num_classes=num_classes,
        num_layers=2,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        use_pos_encoding=True,
    )

    print(f"Q2L created with {num_classes} classes")
    print(f"Num layers: 2")
    print(f"Num heads: 8")
    print(f"Total parameters: {sum(p.numel() for p in q2l.parameters()):,}")

    # Test with sequence features
    seq_features = torch.randn(batch_size, seq_len, feature_dim)
    logits = q2l(seq_features)
    print(f"\nSequence features input: {seq_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    # Test with pooled features
    pooled_features = torch.randn(batch_size, feature_dim)
    logits = q2l(pooled_features)
    print(f"\nPooled features input: {pooled_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    print("\n✓ Q2L test passed!")
