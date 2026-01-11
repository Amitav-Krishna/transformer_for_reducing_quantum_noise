"""
Transformer Autoencoder for density matrix denoising.

Unconstrained output - use post-hoc projection for physical validity.
"""

import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder with unconstrained output.

    Uses ROW-BASED TOKENIZATION: 32 row tokens + 1 [CLS] token for global aggregation.
    Each row token represents one row of the density matrix (32 complex values).

    Architecture:
    - [CLS] token aggregates global information via attention across all rows
    - Decoder uses learnable query with cross-attention to encoder output
    - Global projection from decoder output to full density matrix
    - Row-based tokenization: 1000x fewer attention ops than element-wise

    Output requires post-hoc projection to ensure valid density matrix.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # Row-based tokenization: 32 row tokens + 1 CLS token
        self.num_rows = 32
        self.seq_len = 33  # 32 rows + 1 CLS
        self.input_dim = 64  # 32 complex values = 64 real per row
        self.embed_dim = 64
        self.ffn_dim = 128
        self.num_heads = 8
        self.layers = 5
        self.output_dim = 32 * 32 * 2  # 2048

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)

        # Learned [CLS] token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Positional embeddings for 33 tokens (CLS + 32 rows)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Learnable query for decoder
        self.decoder_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Global projection: decoder output -> full density matrix
        self.global_proj = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x):
        B = x.shape[0]
        # x: (B, 2, 32, 32) -> row tokenization: (B, 32, 64)
        # Each row becomes a token: [real_row, imag_row] concatenated
        real_part = x[:, 0, :, :]  # (B, 32, 32)
        imag_part = x[:, 1, :, :]  # (B, 32, 32)
        row_tokens = torch.cat([real_part, imag_part], dim=-1)  # (B, 32, 64)
        row_tokens = self.input_proj(row_tokens)  # (B, 32, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, row_tokens], dim=1)  # (B, 33, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding  # (B, 33, embed_dim)

        # Encode
        enc = self.encoder(x)

        # Decode with learnable query and cross-attention to encoder output
        query = self.decoder_query.expand(B, -1, -1)  # (B, 1, embed_dim)
        dec = self.decoder(query, enc)  # (B, 1, embed_dim)

        # Global projection to full density matrix
        dec_output = dec[:, 0, :]  # (B, embed_dim)
        out = self.global_proj(dec_output)  # (B, 2048)

        # Reshape to (B, 2, 32, 32)
        out = out.reshape(B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/work/codage/transformer_qnr')
    from losses.frob import FrobeniusFidelityLoss

    model = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
    print(f"TransformerAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
