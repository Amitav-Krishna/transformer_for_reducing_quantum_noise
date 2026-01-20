"""
Count parameters for all model architectures.

Usage:
    python scripts/count_params.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn import CNNAutoencoder
from models.transformer import TransformerAutoencoder
from models.transformer_matched_params import TransformerAutoencoderMatched
from models.transformer_matched_v2 import TransformerAutoencoderMatchedV2
from models.transformer_matched_v3 import TransformerAutoencoderMatchedV3
from models.transformer_matched_v4 import TransformerAutoencoderMatchedV4
from losses.frob import FrobeniusFidelityLoss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    loss_fn = FrobeniusFidelityLoss()

    models = {
        "CNN": CNNAutoencoder(loss_fn=loss_fn),
        "Transformer (small)": TransformerAutoencoder(loss_fn=loss_fn),
        "Transformer Matched v1": TransformerAutoencoderMatched(loss_fn=loss_fn),
        "Transformer Matched v2": TransformerAutoencoderMatchedV2(loss_fn=loss_fn),
        "Transformer Matched v3": TransformerAutoencoderMatchedV3(loss_fn=loss_fn),
        "Transformer Matched v4": TransformerAutoencoderMatchedV4(loss_fn=loss_fn),
    }

    print("Model Parameter Counts")
    print("=" * 50)
    for name, model in models.items():
        params = count_params(model)
        print(f"{name:30s} {params:>12,} params")


if __name__ == "__main__":
    main()