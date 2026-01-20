import torch
import torch.nn as nn


class TransformerAutoencoderMatchedV3(nn.Module):
    """
    Transformer autoencoder with ~750k parameters to match CNN capacity.

    V3 fixes:
    1. Proper FFN ratio: ffn_dim = 4 * embed_dim (standard transformer)
    2. Reduced embed_dim to compensate for larger FFN
    3. Proper bottleneck ratio (50% compression)

    Architecture: embed_dim=76, ffn_dim=304, bottleneck_dim=38, layers=4
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_len = 1024
        self.input_dim = 2
        self.embed_dim = 76
        self.ffn_dim = 304  # 4x embed_dim (standard transformer ratio)
        self.bottleneck_dim = 38  # 50% compression
        self.num_heads = 4
        self.layers = 4

        self.input_proj = nn.Linear(2, self.embed_dim)
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

        # Bottleneck: 76 -> 38 -> 76 (50% compression)
        self.down = nn.Linear(self.embed_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, self.embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        self.output_proj = nn.Linear(self.embed_dim, 2)

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)  # (B,1024,2)
        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)
        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)
        out = self.output_proj(dec)        # (B,1024,2)
        out = out.reshape(B, 32, 32, 2)    # (B,32,32,2)
        out = out.permute(0, 3, 1, 2)      # (B,2,32,32)
        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)