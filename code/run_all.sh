#!/bin/bash
set -e
LOG_DIR="/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea3_coldstart_generalization/data"
CODE_DIR="/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea3_coldstart_generalization/code"
RESULTS_DIR="/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea3_coldstart_generalization/results"

echo "=== Waiting for SMILES fetch to complete ==="
until grep -q "^DONE\." "$LOG_DIR/fetch_smiles.log" 2>/dev/null; do
  sleep 5
done
echo "SMILES fetch complete."

echo "=== Building dataset ==="
python3 "$CODE_DIR/build_dataset.py" > "$LOG_DIR/build_dataset.log" 2>&1
echo "Dataset build complete. Tail:"
tail -20 "$LOG_DIR/build_dataset.log"

echo "=== Running experiments ==="
python3 "$CODE_DIR/run_experiments.py" > "$RESULTS_DIR/run_experiments.log" 2>&1
echo "Experiments complete. Tail:"
tail -40 "$RESULTS_DIR/run_experiments.log"

echo "=== PIPELINE DONE ==="
