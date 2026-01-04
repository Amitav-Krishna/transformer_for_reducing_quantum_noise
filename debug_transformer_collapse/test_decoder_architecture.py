"""
Demonstrate the architectural flaw in decoder(enc, enc).

Shows that using encoder output as BOTH tgt and memory in TransformerDecoder
creates redundant operations.
"""

import torch
import torch.nn as nn

def main():
    print("="*70)
    print("TESTING DECODER ARCHITECTURE FLAW: decoder(enc, enc)")
    print("="*70)

    # Reproduce the transformer architecture
    embed_dim = 64
    num_heads = 8
    ffn_dim = 128
    seq_len = 33
    batch_size = 4

    print(f"\nSetup:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  FFN dim: {ffn_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")

    # Create encoder
    enc_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(enc_layer, num_layers=5)

    # Create decoder
    dec_layer = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        batch_first=True,
    )
    decoder = nn.TransformerDecoder(dec_layer, num_layers=5)

    print("\n" + "-"*70)
    print("CURRENT (FLAWED) ARCHITECTURE")
    print("-"*70)

    # Simulate current architecture
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\nInput: {x.shape}")

    enc = encoder(x)
    print(f"After encoder: {enc.shape}")

    # FLAW: Using enc as both tgt and memory
    dec = decoder(enc, enc)  # This is what the current code does!
    print(f"After decoder(enc, enc): {dec.shape}")

    print("\nWhat happens in decoder(tgt=enc, memory=enc):")
    print("  1. Self-attention: tgt attends to itself (enc attends to enc)")
    print("  2. Cross-attention: result attends to memory (which is ALSO enc)")
    print("  3. This is redundant! We're attending to the same thing twice")

    print("\n" + "-"*70)
    print("WHAT A PROPER AUTOENCODER SHOULD DO")
    print("-"*70)

    print("\nOption 1: Use encoder-only architecture")
    print("  encoder(x) -> bottleneck -> encoder(bottleneck) -> output")
    print("  OR: Use cross-attention with learnable queries")

    print("\nOption 2: Use decoder with learned queries")
    print("  enc = encoder(x)")
    print("  queries = learned_queries.expand(batch_size, -1, -1)")
    print("  dec = decoder(tgt=queries, memory=enc)")
    print("  This way decoder learns to extract information from encoder via queries")

    print("\nOption 3: Use two separate encoders")
    print("  enc = encoder(x)")
    print("  dec = decoder_encoder(enc)  # Another TransformerEncoder")

    print("\n" + "-"*70)
    print("WHY CURRENT ARCHITECTURE FAILS")
    print("-"*70)

    print("\nThe current architecture has:")
    print("  ❌ No bottleneck (33 tokens -> 33 tokens)")
    print("  ❌ Redundant attention (self-attn + cross-attn to same thing)")
    print("  ❌ No forced information compression")
    print("  ❌ Model can learn identity mapping too easily")

    print("\nResult:")
    print("  → Model learns to output uniform/average of all training data")
    print("  → For quantum density matrices, this average IS the maximally mixed state")
    print("  → Output: I/32 regardless of input")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nReplace decoder(enc, enc) with one of:")
    print("  1. Learned query tokens that cross-attend to encoder")
    print("  2. Explicit bottleneck + second encoder")
    print("  3. Switch to encoder-only architecture with proper bottleneck")

if __name__ == "__main__":
    main()
