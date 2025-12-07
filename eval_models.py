import torch
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.evaluate_all_on_test import evaluate_all_checkpoints_on_test

from train_models import EXPERIMENTS
import models.transformer
print(">>> Transformer loaded from:", models.transformer.__file__)

import sys
print(">>> Python path:", sys.path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset")
    chunks = load_chunks("dataset_smaller")

    # Same seed, same test set
    _, _, test_chunks = split_chunks(chunks, seed=42)

    print("Starting full checkpoint evaluation...")
    results = evaluate_all_checkpoints_on_test(
        EXPERIMENTS,
        device,
        test_chunks
        )

    print("\n==== DONE ====")
    print("Test losses by model:")
    for model_name, model_results in results.items():
        print(model_name, ":", model_results)

if __name__ == "__main__":
    main()
