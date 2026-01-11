#!/bin/bash
# Queue: 8-qubit dataset generation → 5-qubit hierarchical training → 8-qubit hierarchical training

cd /workspace
source /workspace/venv/bin/activate

# Wait for 8-qubit dataset generation to complete
echo "Waiting for 8-qubit dataset generation to complete..."
while ps aux | grep -v grep | grep "generate_8qubit_dataset.py" > /dev/null; do
    sleep 60
done
echo "8-qubit dataset generation complete!"

# Upload 5-qubit dataset if not present
if [ ! -d "/workspace/dataset_smaller" ]; then
    echo "5-qubit dataset not found, please upload it first"
    exit 1
fi

# Train hierarchical transformer on 5 qubits
echo "Starting 5-qubit hierarchical transformer training..."
python /workspace/train_hierarchical_5qubit.py > /workspace/train_hierarchical_5qubit.log 2>&1
echo "5-qubit hierarchical training complete!"

# Train hierarchical transformer on 8 qubits (will be created next)
if [ -f "/workspace/train_hierarchical_8qubit.py" ]; then
    echo "Starting 8-qubit hierarchical transformer training..."
    python /workspace/train_hierarchical_8qubit.py > /workspace/train_hierarchical_8qubit.log 2>&1
    echo "8-qubit hierarchical training complete!"
fi

echo "All jobs complete!"
