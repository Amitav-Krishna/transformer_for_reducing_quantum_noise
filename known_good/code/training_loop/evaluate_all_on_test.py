import os
import torch
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks
from training_loop.fidelity import evaluate_fidelity_on_chunks


def evaluate_all_checkpoints_on_test(EXPERIMENTS, device, test_chunks):
    """
    Evaluate *every epoch checkpoint* for every model on the test set.
    Logs BOTH loss and fidelity to csvs_2/{model_name}_test.csv.
    Returns nested dict.
    """

    results = {}

    for name, config in EXPERIMENTS.items():
        arch = config["arch"]
        ckpt_dir = f"checkpoints_2/{name}"

        if not os.path.exists(ckpt_dir):
            print(f"[WARNING] No checkpoint directory for {name}, skipping.")
            continue

        print(f"\n===== Evaluating ALL checkpoints for {name} =====")

        csv_logger = CSVLogger("csvs_2", f"{name}_test")

        model_results = {}

        ckpts = sorted(
            f for f in os.listdir(ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        )

        for ckpt_file in ckpts:
            epoch = int(ckpt_file.split("_")[1].split(".")[0])
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)

            print(f"Evaluating {name} epoch {epoch} ...")

            model = config["create_model"]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            BATCH = 32 if arch == "cnn" else 8

            # LOSS
            test_loss = evaluate_on_chunks(model, test_chunks, arch, device, BATCH)

            # FIDELITY
            fid_mean, fid_std = evaluate_fidelity_on_chunks(
                model, test_chunks, arch, device, BATCH
            )

            print(f"  Test Loss: {test_loss:.6f} | Fidelity: {fid_mean:.4f}")

            # CSV logging: patch our logger to include fidelity columns
            csv_logger.log_val(epoch=epoch, val_loss=test_loss)
            # Append a fidelity-only line:
            with open(csv_logger.csv_path, "a") as f:
                f.write(f"{epoch},fid,,,{fid_mean}\n")

            model_results[epoch] = {
                "loss": float(test_loss),
                "fid_mean": float(fid_mean),
                "fid_std": float(fid_std),
            }

        results[name] = model_results

    return results
