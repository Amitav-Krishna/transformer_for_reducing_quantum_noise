import os
import torch
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from torch.utils.data import DataLoader


def evaluate_all_checkpoints_on_test(EXPERIMENTS, device, test_chunks):
    """
    Evaluate *every epoch checkpoint* for every model on the test set.
    Logs results to csvs_2/{model_name}_test.csv.
    Returns a dict: {model_name: {epoch: test_loss}}
    """

    results = {}

    for name, config in EXPERIMENTS.items():
        arch = config["arch"]
        ckpt_dir = f"checkpoints_2/{name}"

        if not os.path.exists(ckpt_dir):
            print(f"[WARNING] No checkpoint directory for {name}, skipping.")
            continue

        print(f"\n===== Evaluating ALL checkpoints for {name} =====")

        # Logger for this model's test results
        csv_logger = CSVLogger("csvs_2", f"{name}_test")

        # Store test results in-memory too
        model_results = {}

        # Evaluate in order of epochs
        ckpts = sorted(
            f for f in os.listdir(ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        )

        for ckpt_file in ckpts:
            epoch = int(ckpt_file.split("_")[1].split(".")[0])
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)

            print(f"Evaluating {name} epoch {epoch} ...")

            # Recreate and load model
            model = config["create_model"]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            BATCH = 32 if arch == "cnn" else 8

            # Compute test loss
            test_loss = evaluate_on_chunks(model, test_chunks, arch, device, BATCH)
            print(f"  Test Loss: {test_loss:.6f}")

            # Log to CSV
            csv_logger.log_val(epoch=epoch, val_loss=float(test_loss))

            # Store in-memory
            model_results[epoch] = float(test_loss)

        # Also evaluate BEST checkpoint last
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            print("Evaluating BEST checkpoint ...")
            model = config["create_model"]().to(device)
            model.load_state_dict(torch.load(best_path, map_location=device))
            BATCH = 32 if arch == "cnn" else 8
            best_loss = evaluate_on_chunks(model, test_chunks, arch, device, BATCH)
            csv_logger.log_val(epoch="BEST", val_loss=float(best_loss))
            model_results["BEST"] = float(best_loss)

        results[name] = model_results

    return results
