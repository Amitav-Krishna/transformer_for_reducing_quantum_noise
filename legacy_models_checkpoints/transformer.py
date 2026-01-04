import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_len = 1024
        self.input_dim = 2
        self.embed_dim = 32
        self.ffn_dim = 64
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

        self.down = nn.Linear(self.embed_dim, 16)
        self.up   = nn.Linear(16, self.embed_dim)

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
        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)
        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)
        return self.output_proj(dec)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)

