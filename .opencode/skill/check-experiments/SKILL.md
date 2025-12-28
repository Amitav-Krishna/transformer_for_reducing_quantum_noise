---
name: check-experiments
description: Check the status of running experiments on RunPod. Use when the user asks about experiment progress, training status, GPU usage, or wants to monitor the train_16 pipeline.
---

# Check Experiments Skill

Monitor running experiments on the RunPod GPU server.

## IMPORTANT: Always Use the Status Script First

**Run this command first to get a comprehensive status overview:**

```bash
python .opencode/skill/check-experiments/scripts/status.py
```

This script provides everything you need in one command:
- Pipeline running status and current phase
- Progress percentage with rate and ETA
- GPU memory, utilization, temperature, and power
- Disk usage
- Dataset generation progress (chunk counts)
- Checkpoint status (which models are trained)
- Training CSV logs
- Time estimates for remaining work
- Recent errors/warnings
- Last 15 lines of log output

**Only use manual SSH commands if the status script fails or you need specific debugging information.**

## Important Limitations

- **Only the user can access the RunPod dashboard** - do not attempt to access any web interfaces, dashboards, or control panels
- If SSH connection fails, ask the user to check the RunPod dashboard for the current port
- All checks must be done via SSH commands only

## Connection Details

- **Host**: root@157.157.221.29
- **SSH Key**: ~/.ssh/id_ed25519
- **Port**: 20610 (may change if storage is modified)
- **Workspace**: /workspace
- **Venv**: source /workspace/venv/bin/activate

## Manual SSH Commands (Fallback Only)

Use these only if the status script fails or for specific debugging:

### Check if pipeline is running
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 -o ConnectTimeout=10 root@157.157.221.29 "ps aux | grep -E 'queue_float64|python' | grep -v grep | head -10"
```

### View recent log output
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 -o ConnectTimeout=10 root@157.157.221.29 "tail -50 /workspace/train_16_run.log"
```

### Check GPU usage
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 -o ConnectTimeout=10 root@157.157.221.29 "nvidia-smi"
```

### Check disk space
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 -o ConnectTimeout=10 root@157.157.221.29 "df -h /workspace"
```

### Check for errors
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 -o ConnectTimeout=10 root@157.157.221.29 "grep -i 'error\|exception' /workspace/train_16_run.log | tail -20"
```

## Pipeline Phases

The train_16 float64 pipeline runs in this order:

1. **Phase 1**: Generate 5-qubit float64 dataset (~2 hours)
   - Output: `/workspace/dataset_5qubit_float64/`
   - 100,000 samples at ~14-30 samples/sec

2. **Phase 2**: Train 5-qubit models (~12-16 hours)
   - transformer_5qubit (~1.09M params)
   - mlp_5qubit (~1.09M params) 
   - mlp_5qubit_wide (~5.29M params)
   - mlp_5qubit_deep (~2.15M params)
   - Each model: 100 epochs with Uhlmann evaluation after

3. **Phase 3**: Generate 8-qubit float64 dataset (~8-10 hours)
   - Output: `/workspace/dataset_8qubit_float64/`
   - 100,000 samples (256x256 matrices)

4. **Phase 4**: Train 8-qubit models (~20+ hours)
   - transformer_8qubit (~1.61M params)
   - mlp_8qubit (~1.61M params)

## Troubleshooting

### SSH Connection Failed
The port may have changed. **IMPORTANT: Only the user can check the RunPod dashboard** - do not attempt to access any web dashboards or RunPod control panels. If SSH fails, inform the user that the port may have changed and ask them to check the RunPod dashboard for the current port number.

### Pipeline Crashed
Check the log for errors:
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 root@157.157.221.29 "grep -i error /workspace/train_16_run.log | tail -20"
```

### Restart Pipeline
```bash
ssh -i ~/.ssh/id_ed25519 -p 20610 root@157.157.221.29 "cd /workspace && source venv/bin/activate && nohup bash train_16/queue_float64.sh > train_16_run.log 2>&1 &"
```

## Results Location

After completion:
- Checkpoints: `/workspace/train_16/checkpoints_16/<model_name>/best.pt`
- Training logs: `/workspace/train_16/csvs_16/<model_name>_*.csv`
- VRAM logs: `/workspace/train_16/csvs_16/<model_name>_vram.csv`
