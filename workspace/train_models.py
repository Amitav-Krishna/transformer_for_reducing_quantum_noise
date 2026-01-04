import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


###############################################################################
# Dataset Loader
###############################################################################

from training_loop.dataset.ChunkDataset import ChunkDataset


###############################################################################
# Import unified canonical models + losses
###############################################################################

from models.cnn import CNNAutoencoder
from models.transformer import TransformerAutoencoder
from models.transformer_matched_params import TransformerAutoencoderMatched
from losses.frob import FrobeniusFidelityLoss
from losses.physics import DensityMatrixPhysicsLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss



###############################################################################
# Utility: Load dataset
###############################################################################

from training_loop.dataset.load_chunks import load_chunks


###############################################################################
# Model configurations (all 4 experiments)
###############################################################################

EXPERIMENTS = {
        "cnn_frob": {
            "arch": "cnn",
            "loss": "frob",
            "create_model": lambda: CNNAutoencoder(loss_fn=FrobeniusFidelityLoss())
        },

    "cnn_physics": {
        "arch": "cnn",
        "loss": "physics",
        "create_model": lambda: CNNAutoencoder(loss_fn=CompositePhysicsTotalLoss())
    },

    "transformer_frob": {
        "arch": "transformer",
        "loss": "frob",
        "create_model": lambda: TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
    },

    "transformer_physics": {
        "arch": "transformer",
        "loss": "physics",
        "create_model": lambda: TransformerAutoencoder(loss_fn=CompositePhysicsTotalLoss())
    },

    "transformer_matched_frob": {
        "arch": "transformer",
        "loss": "frob",
        "create_model": lambda: TransformerAutoencoderMatched(loss_fn=FrobeniusFidelityLoss())
    },

    "transformer_matched_physics": {
        "arch": "transformer",
        "loss": "physics",
        "create_model": lambda: TransformerAutoencoderMatched(loss_fn=CompositePhysicsTotalLoss())
    },
}


# Train / val / split

from training_loop.dataset.split_chunks import split_chunks


###############################################################################
# Training routine for one experiment
###############################################################################

from training_loop.train_single_experiment import train_single_experiment

 
###############################################################################
# MAIN: Train all four models sequentially
###############################################################################

from training_loop.evaluate_all_on_test import evaluate_all_checkpoints_on_test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)
    
    for name, config in EXPERIMENTS.items():
        if name != "transformer_physics":
            # already ran training
            continue 
        train_single_experiment(name, config, train_chunks, val_chunks, device)

    test_results = evaluate_all_checkpoints_on_test(EXPERIMENTS, test_chunks, device)

if __name__ == "__main__":
    main()

