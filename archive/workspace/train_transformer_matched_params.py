import torch

from models.transformer_matched_params import TransformerAutoencoderMatched
from losses.frob import FrobeniusFidelityLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.train_single_experiment import train_single_experiment


EXPERIMENTS = {
    "transformer_matched_frob": {
        "arch": "transformer",
        "loss": "frob",
        "create_model": lambda: TransformerAutoencoderMatched(
            loss_fn=FrobeniusFidelityLoss()
        ),
    },
    "transformer_matched_physics": {
        "arch": "transformer",
        "loss": "physics",
        "create_model": lambda: TransformerAutoencoderMatched(
            loss_fn=CompositePhysicsTotalLoss()
        ),
    },
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Verify parameter count
    model = TransformerAutoencoderMatched()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerAutoencoderMatched params: {params:,}")
    print(f"(Target: ~750k to match CNN's 748,898)")
    del model

    # Load dataset with same split as original experiments
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks, seed=42)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")
    print(f"Test chunks:  {len(test_chunks)}")

    for name, config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        train_single_experiment(name, config, train_chunks, val_chunks, device)

    print("\nDone. Checkpoints saved to checkpoints_2/transformer_matched_*")


if __name__ == "__main__":
    main()
