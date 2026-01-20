#!/bin/bash

set -e

SRC="dataset"
DST="dataset_smaller"

mkdir -p "$DST"

# Noise types and levels
types=("depolarizing" "amplitude_damping" "phase_damping" "bitflip" "mixed")
levels=("0.05" "0.1" "0.15" "0.2")

# We need 50 files per (type, level) cell
FILES_PER_CELL=50

echo "Creating balanced dataset in $DST ..."
echo

for t in "${types[@]}"; do
  for l in "${levels[@]}"; do

    echo "Processing $t $l ..."

    # List matching files, sorted
    matches=( $(ls "$SRC" | grep "^${t}_${l}_" | sort) )

    # Ensure enough files exist
    if [ ${#matches[@]} -lt $FILES_PER_CELL ]; then
      echo "ERROR: Not enough files for $t $l. Found ${#matches[@]}, need $FILES_PER_CELL."
    fi

    # Copy first N files into dataset_smaller
    for ((i=0; i<FILES_PER_CELL; i++)); do
      cp "$SRC/${matches[$i]}" "$DST/"
    done

    echo "  Copied $FILES_PER_CELL files for $t $l."

  done
done

echo
echo "Done! Balanced dataset created at $DST"
echo "Total files copied:"
ls "$DST" | wc -l

