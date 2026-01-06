"""
ML-Decoder: Scalable and Versatile Classification Head

Based on the paper: "ML-Decoder: Scalable and Versatile Classification Head"
https://arxiv.org/abs/2111.12933

Official implementation: https://github.com/Alibaba-MIIL/ML_Decoder
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class MLDecoder(nn.Module):
    """
    Multi-Label Decoder for image classification.

    Uses learnable query embeddings and cross-attention to model
    class dependencies and improve multi-label classification performance.
    """

    def __init__(
        self,
        num_classes: int,
        initial_num_features: int,
        num_queries: int = None,
        num_layers: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = False,
        return_intermediate: bool = False,
    ):
        """
        Args:
            num_classes (int): Number of output classes
            initial_num_features (int): Input feature dimension (from vision encoder)
            num_queries (int, optional): Number of query embeddings. If None, uses num_classes
            num_layers (int): Number of transformer decoder layers
            num_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
            normalize_before (bool): Whether to normalize before or after attention
            return_intermediate (bool): Whether to return intermediate layer outputs
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries if num_queries is not None else num_classes
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # Query embeddings (learnable)
        self.query_embed = nn.Embedding(self.num_queries, initial_num_features)

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=initial_num_features,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=normalize_before,
        )

        decoder_norm = nn.LayerNorm(initial_num_features)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=decoder_norm,
            return_intermediate=return_intermediate,
        )

        # Output projection
        self.classifier = nn.Linear(initial_num_features, num_classes)

        # Group linear (optional feature grouping, disabled by default)
        self.group_linear = None

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features from vision encoder
                       Shape: (batch_size, num_features) for pooled features
                       or (batch_size, seq_len, num_features) for sequence features

        Returns:
            Tensor: Class logits, shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Handle both pooled and sequence features
        if x.dim() == 2:
            # Pooled features: (batch_size, num_features)
            # Expand to sequence: (batch_size, 1, num_features)
            x = x.unsqueeze(1)

        # x: (batch_size, seq_len, num_features)
        # Transpose for decoder: (seq_len, batch_size, num_features)
        memory = x.transpose(0, 1)

        # Query embeddings: (num_queries, num_features)
        # Expand for batch: (num_queries, batch_size, num_features)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Decoder forward
        # Output: (num_queries, batch_size, num_features)
        hs = self.decoder(query_embed, memory)

        # Get final layer output
        if self.return_intermediate:
            hs = hs[-1]  # Last layer

        # Transpose back: (batch_size, num_queries, num_features)
        hs = hs.transpose(0, 1)

        # Average pool over queries: (batch_size, num_features)
        # Or use only the first query, or max pool
        # Here we use average pooling
        hs = hs.mean(dim=1)

        # Classification head
        logits = self.classifier(hs)

        return logits


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()

        self.normalize_before = normalize_before

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Query embeddings (num_queries, batch_size, d_model)
            memory: Encoder output (seq_len, batch_size, d_model)
        """
        # Self-attention
        if self.normalize_before:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # Cross-attention
        if self.normalize_before:
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.cross_attn(tgt2, memory, memory, attn_mask=memory_mask)
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # Feedforward
        if self.normalize_before:
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers"""

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            decoder_layer for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output)

        if self.norm is not None and not self.return_intermediate:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


if __name__ == "__main__":
    # Test ML Decoder
    print("=== Testing ML Decoder ===")

    batch_size = 4
    num_classes = 5
    feature_dim = 768  # SigLIP base hidden size

    # Create ML Decoder
    decoder = MLDecoder(
        num_classes=num_classes,
        initial_num_features=feature_dim,
        num_layers=1,
        num_heads=8,
    )

    print(f"ML Decoder created with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test with pooled features
    pooled_features = torch.randn(batch_size, feature_dim)
    logits = decoder(pooled_features)
    print(f"\nPooled features input: {pooled_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    # Test with sequence features
    seq_len = 196  # 14x14 patches for 224x224 image
    seq_features = torch.randn(batch_size, seq_len, feature_dim)
    logits = decoder(seq_features)
    print(f"\nSequence features input: {seq_features.shape}")
    print(f"Output logits: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)

    print("\n✓ ML Decoder test passed!")
