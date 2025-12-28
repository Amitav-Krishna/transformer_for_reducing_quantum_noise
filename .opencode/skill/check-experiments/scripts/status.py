#!/usr/bin/env python3
"""
Check the status of running experiments on RunPod.

This script SSHs to RunPod and checks:
- Whether the pipeline is running
- Current phase and progress
- GPU memory usage
- Disk space
- Dataset and checkpoint status
- Recent log output with time estimates
"""

import subprocess
import sys
import re
from datetime import datetime, timedelta

# RunPod connection details
SSH_KEY = "~/.ssh/id_ed25519"
SSH_PORT = "20610"
SSH_HOST = "root@157.157.221.29"
SSH_OPTS = (
    f"-i {SSH_KEY} -p {SSH_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
)

# Expected times for each phase (in hours)
PHASE_TIMES = {
    "phase1": 2.0,  # 5-qubit dataset generation
    "phase2": 14.0,  # 5-qubit training (4 models)
    "phase3": 9.0,  # 8-qubit dataset generation
    "phase4": 22.0,  # 8-qubit training (2 models)
}

# Models in training order
PHASE2_MODELS = [
    "transformer_5qubit",
    "mlp_5qubit",
    "mlp_5qubit_wide",
    "mlp_5qubit_deep",
]
PHASE4_MODELS = ["transformer_8qubit", "mlp_8qubit"]


def ssh_cmd(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command and return (success, output)."""
    full_cmd = f'ssh {SSH_OPTS} {SSH_HOST} "{cmd}"'
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "SSH command timed out"
    except Exception as e:
        return False, str(e)


def check_connection() -> bool:
    """Test SSH connection."""
    success, _ = ssh_cmd("echo OK")
    return success


def check_pipeline_running() -> tuple[bool, str, str]:
    """Check if the pipeline is running and what it's doing.
    Returns (is_running, status_message, current_phase)
    """
    success, output = ssh_cmd(
        "ps aux | grep -E 'queue_float64|python.*train_16' | grep -v grep"
    )
    if not success or not output.strip():
        return False, "Pipeline not running", ""

    # Parse what's running
    if "generate_5qubit" in output:
        return True, "Phase 1: Generating 5-qubit dataset", "phase1"
    elif "generate_8qubit" in output:
        return True, "Phase 3: Generating 8-qubit dataset", "phase3"
    elif "train_transformer_5qubit" in output:
        return True, "Phase 2: Training transformer_5qubit (1/4)", "phase2"
    elif "train_mlp_5qubit_wide" in output:
        return True, "Phase 2: Training mlp_5qubit_wide (3/4)", "phase2"
    elif "train_mlp_5qubit_deep" in output:
        return True, "Phase 2: Training mlp_5qubit_deep (4/4)", "phase2"
    elif "train_mlp_5qubit" in output and "wide" not in output and "deep" not in output:
        return True, "Phase 2: Training mlp_5qubit (2/4)", "phase2"
    elif "train_transformer_8qubit" in output:
        return True, "Phase 4: Training transformer_8qubit (1/2)", "phase4"
    elif "train_mlp_8qubit" in output:
        return True, "Phase 4: Training mlp_8qubit (2/2)", "phase4"
    elif "eval_uhlmann" in output:
        return True, "Evaluating model on Uhlmann fidelity", "eval"
    elif "queue_float64" in output:
        return True, "Pipeline running (between tasks)", "unknown"

    return True, "Pipeline running (unknown task)", "unknown"


def get_progress_details() -> dict:
    """Get detailed progress from log file."""
    success, output = ssh_cmd("tail -200 /workspace/train_16_run.log 2>/dev/null")

    result = {
        "raw_progress": None,
        "percentage": None,
        "current": None,
        "total": None,
        "rate": None,
        "eta": None,
        "epoch": None,
        "total_epochs": None,
        "phase_start": None,
    }

    if not success:
        return result

    lines = output.strip().split("\n")

    # Look for phase start times
    for line in lines:
        if (
            "Phase 1:" in line
            or "Phase 2:" in line
            or "Phase 3:" in line
            or "Phase 4:" in line
        ):
            # Try to extract timestamp
            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                result["phase_start"] = match.group(1)

    # Look for progress indicators (scan in reverse for most recent)
    for line in reversed(lines):
        # Match tqdm patterns like "45%|████      | 45000/100000 [1:23:45<0:45:30, 14.5it/s]"
        match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", line)
        if match:
            result["percentage"] = int(match.group(1))
            result["current"] = int(match.group(2))
            result["total"] = int(match.group(3))

            # Parse the time info: "1:23:45<0:45:30, 14.5it/s"
            time_info = match.group(4)
            eta_match = re.search(r"<([^,]+)", time_info)
            rate_match = re.search(r"(\d+\.?\d*)\s*it/s", time_info)

            if eta_match:
                result["eta"] = eta_match.group(1).strip()
            if rate_match:
                result["rate"] = float(rate_match.group(1))

            result["raw_progress"] = (
                f"{result['percentage']}% ({result['current']}/{result['total']})"
            )
            break

        # Match epoch progress like "Epoch 45/100"
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
        if epoch_match:
            result["epoch"] = int(epoch_match.group(1))
            result["total_epochs"] = int(epoch_match.group(2))
            result["percentage"] = int(100 * result["epoch"] / result["total_epochs"])
            result["raw_progress"] = f"Epoch {result['epoch']}/{result['total_epochs']}"
            break

    return result


def get_gpu_status() -> dict:
    """Get GPU memory usage and utilization."""
    success, output = ssh_cmd(
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits"
    )

    result = {
        "memory_used": None,
        "memory_total": None,
        "utilization": None,
        "temperature": None,
        "power": None,
        "formatted": "Could not query GPU",
    }

    if not success:
        return result

    try:
        parts = output.strip().split(",")
        if len(parts) >= 3:
            result["memory_used"] = int(parts[0].strip())
            result["memory_total"] = int(parts[1].strip())
            result["utilization"] = int(parts[2].strip())
            if len(parts) >= 4:
                result["temperature"] = int(parts[3].strip())
            if len(parts) >= 5:
                try:
                    result["power"] = float(parts[4].strip())
                except:
                    pass

            mem_pct = 100 * result["memory_used"] / result["memory_total"]
            result["formatted"] = (
                f"{result['memory_used']}MB / {result['memory_total']}MB ({mem_pct:.0f}% memory), "
                f"{result['utilization']}% GPU util"
            )
            if result["temperature"]:
                result["formatted"] += f", {result['temperature']}°C"
            if result["power"]:
                result["formatted"] += f", {result['power']:.0f}W"
    except:
        pass

    return result


def get_disk_status() -> dict:
    """Get disk space."""
    success, output = ssh_cmd("df -h /workspace | tail -1")

    result = {
        "used": None,
        "total": None,
        "percent": None,
        "formatted": "Could not query disk",
    }

    if not success:
        return result

    try:
        parts = output.split()
        if len(parts) >= 5:
            result["total"] = parts[1]
            result["used"] = parts[2]
            result["percent"] = parts[4]
            result["formatted"] = (
                f"{result['used']} used / {result['total']} ({result['percent']} full)"
            )
    except:
        pass

    return result


def get_datasets() -> list[dict]:
    """Check which datasets exist with chunk counts."""
    success, output = ssh_cmd(
        "for d in /workspace/dataset_*_float64; do "
        'if [ -d "$d" ]; then '
        'echo "$d $(ls $d 2>/dev/null | wc -l)"; '
        "fi; done"
    )

    datasets = []
    if not success or not output.strip():
        return datasets

    for line in output.strip().split("\n"):
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                path = parts[0].replace("/workspace/", "")
                count = int(parts[1])
                # Estimate samples (1000 samples per chunk typically)
                samples = count * 1000
                datasets.append({"name": path, "chunks": count, "samples_est": samples})

    return datasets


def get_checkpoints() -> list[dict]:
    """Check which checkpoints exist with file sizes."""
    success, output = ssh_cmd(
        "find /workspace/train_16/checkpoints_16 -name '*.pt' -exec ls -lh {} \\; 2>/dev/null"
    )

    checkpoints = []
    if not success or not output.strip():
        return checkpoints

    for line in output.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if len(parts) >= 9:
                size = parts[4]
                path = parts[8]
                # Extract model name
                path_parts = path.split("/")
                model = path_parts[-2] if len(path_parts) >= 2 else "unknown"
                filename = path_parts[-1]
                checkpoints.append({"model": model, "file": filename, "size": size})

    return checkpoints


def get_csv_logs() -> list[dict]:
    """Check which CSV training logs exist."""
    success, output = ssh_cmd("ls -la /workspace/train_16/csvs_16/*.csv 2>/dev/null")

    csvs = []
    if not success or not output.strip():
        return csvs

    for line in output.strip().split("\n"):
        if ".csv" in line:
            parts = line.split()
            if len(parts) >= 9:
                size = parts[4]
                filename = parts[8].split("/")[-1]
                csvs.append({"file": filename, "size": size})

    return csvs


def get_recent_log(lines: int = 10) -> str:
    """Get last few lines of log, handling tqdm single-line output."""
    # Get the header info (first 13 lines are usually the header before tqdm starts)
    success, header_output = ssh_cmd("head -13 /workspace/train_16_run.log 2>/dev/null")

    # Also get the tail to extract the latest tqdm progress bar
    success_tail, tail_output = ssh_cmd(
        "tail -c 5000 /workspace/train_16_run.log 2>/dev/null"
    )

    result_lines = []

    # Add header lines (skip very long ones)
    if success and header_output.strip():
        for line in header_output.strip().split("\n"):
            if len(line) <= 200:
                result_lines.append(line)

    # Extract the last tqdm progress bar from tail
    # tqdm lines look like: "  X%|████      | 12345/100000 [1:23:45<0:45:30, 14.5it/s]"
    if success_tail and tail_output:
        # Find all tqdm progress entries by looking for the pattern with brackets
        # Each tqdm update has a pattern like "X%|...|" followed by "[time<eta, rate]"
        import re

        # Find the last complete tqdm entry (ends with "]" or "it/s]")
        matches = list(
            re.finditer(r"\s*(\d+%\|[^|]+\|\s*\d+/\d+\s*\[[^\]]+\])", tail_output)
        )
        if matches:
            last_progress = matches[-1].group(1).strip()
            result_lines.append("")
            result_lines.append("Latest progress:")
            result_lines.append(f"  {last_progress}")

    if not result_lines:
        return "Could not read log"

    return "\n".join(result_lines[-lines:])


def get_errors() -> list[str]:
    """Check for recent errors in log."""
    success, output = ssh_cmd(
        "grep -i 'error\\|exception\\|failed\\|killed' /workspace/train_16_run.log 2>/dev/null | tail -5"
    )
    if not success or not output.strip():
        return []
    return [line.strip() for line in output.strip().split("\n") if line.strip()]


def format_time_remaining(phase: str, progress: dict, completed_models: list) -> str:
    """Estimate time remaining based on current phase and progress."""
    if not phase:
        return "Unknown"

    estimates = []

    # Current phase estimate
    if progress.get("eta"):
        estimates.append(f"Current task: ~{progress['eta']}")
    elif progress.get("percentage") and progress["percentage"] > 0:
        # Rough estimate based on percentage
        if phase == "phase1":
            remaining_pct = 100 - progress["percentage"]
            est_hours = PHASE_TIMES["phase1"] * remaining_pct / 100
            estimates.append(f"Phase 1: ~{est_hours:.1f}h remaining")
        elif phase == "phase3":
            remaining_pct = 100 - progress["percentage"]
            est_hours = PHASE_TIMES["phase3"] * remaining_pct / 100
            estimates.append(f"Phase 3: ~{est_hours:.1f}h remaining")

    # Remaining phases
    remaining_hours = 0
    if phase == "phase1":
        remaining_hours = (
            PHASE_TIMES["phase2"] + PHASE_TIMES["phase3"] + PHASE_TIMES["phase4"]
        )
    elif phase == "phase2":
        # Estimate based on models completed
        models_done = len([c for c in completed_models if "5qubit" in c])
        models_remaining = 4 - models_done
        remaining_hours = (
            (PHASE_TIMES["phase2"] * models_remaining / 4)
            + PHASE_TIMES["phase3"]
            + PHASE_TIMES["phase4"]
        )
    elif phase == "phase3":
        remaining_hours = PHASE_TIMES["phase4"]
    elif phase == "phase4":
        models_done = len([c for c in completed_models if "8qubit" in c])
        models_remaining = 2 - models_done
        remaining_hours = PHASE_TIMES["phase4"] * models_remaining / 2

    if remaining_hours > 0:
        estimates.append(f"Total remaining: ~{remaining_hours:.0f}h")

    return " | ".join(estimates) if estimates else "Unknown"


def main():
    print("=" * 70)
    print("EXPERIMENT STATUS - train_16 Float64 Pipeline")
    print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Check connection
    print("Connecting to RunPod...")
    if not check_connection():
        print("  ERROR: Could not connect to RunPod!")
        print("  - Check if RunPod instance is running")
        print(f"  - Current port: {SSH_PORT} (may have changed)")
        print("  - Ask user to check RunPod dashboard for new port")
        sys.exit(1)
    print("  Connected!")
    print()

    # Check pipeline status
    running, status, phase = check_pipeline_running()
    status_icon = "🟢" if running else "🔴"
    print(f"Pipeline Status: {status_icon} {'RUNNING' if running else 'NOT RUNNING'}")
    print(f"  Current: {status}")

    # Get detailed progress
    progress = get_progress_details()
    if progress["raw_progress"]:
        print(f"  Progress: {progress['raw_progress']}")
        if progress["rate"]:
            print(f"  Rate: {progress['rate']:.1f} it/s")
        if progress["eta"]:
            print(f"  ETA: {progress['eta']}")
    print()

    # GPU status
    gpu = get_gpu_status()
    gpu_icon = (
        "🔥"
        if gpu.get("utilization", 0) > 80
        else "💤"
        if gpu.get("utilization", 0) < 5
        else "⚡"
    )
    print(f"GPU: {gpu_icon} {gpu['formatted']}")

    # Disk status
    disk = get_disk_status()
    print(f"Disk: {disk['formatted']}")
    print()

    # Datasets
    datasets = get_datasets()
    print("Datasets:")
    if datasets:
        for ds in datasets:
            status = "✓" if ds["chunks"] >= 100 else "⏳"
            print(
                f"  {status} {ds['name']}: {ds['chunks']} chunks (~{ds['samples_est']:,} samples)"
            )
    else:
        print("  No float64 datasets found yet")
    print()

    # Checkpoints
    checkpoints = get_checkpoints()
    completed_models = list(
        set(c["model"] for c in checkpoints if c["file"] == "best.pt")
    )
    print("Checkpoints:")
    if checkpoints:
        # Group by model
        by_model = {}
        for cp in checkpoints:
            if cp["model"] not in by_model:
                by_model[cp["model"]] = []
            by_model[cp["model"]].append(cp)

        for model, cps in by_model.items():
            has_best = any(c["file"] == "best.pt" for c in cps)
            status = "✓" if has_best else "⏳"
            files = ", ".join(f"{c['file']} ({c['size']})" for c in cps)
            print(f"  {status} {model}: {files}")
    else:
        print("  No checkpoints yet")
    print()

    # CSV logs
    csvs = get_csv_logs()
    if csvs:
        print(f"Training Logs: {len(csvs)} CSV files")
        for csv in csvs[:5]:  # Show first 5
            print(f"  - {csv['file']} ({csv['size']} bytes)")
        if len(csvs) > 5:
            print(f"  ... and {len(csvs) - 5} more")
        print()

    # Time estimate
    time_est = format_time_remaining(phase, progress, completed_models)
    print(f"Time Estimate: {time_est}")
    print()

    # Check for errors
    errors = get_errors()
    if errors:
        print("⚠️  Recent Errors/Warnings:")
        for err in errors:
            print(f"  {err[:100]}")
        print()

    # Recent log
    print("-" * 70)
    print("RECENT LOG OUTPUT (last 10 lines):")
    print("-" * 70)
    print(get_recent_log(10))
    print()


if __name__ == "__main__":
    main()
