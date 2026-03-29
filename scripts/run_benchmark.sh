#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results plots

make

OUT_CSV="results/benchmark.csv"
rm -f "$OUT_CSV"

SEEDS=(101 202 303 404 505)
for seed in "${SEEDS[@]}"; do
  echo "Running seed ${seed}..."
  ./bin/train \
    --model all \
    --data data/tinyshakespeare.txt \
    --epochs 4 \
    --steps 700 \
    --ctx 32 \
    --dmodel 16 \
    --hidden 24 \
    --lr 0.02 \
    --seed "$seed" \
    --benchmark "$OUT_CSV"
done

python3 scripts/summarize_results.py "$OUT_CSV" results/summary.csv results/summary.md
python3 scripts/plot_results.py "$OUT_CSV" plots/val_loss.svg plots/runtime.svg

echo "Benchmark complete."
echo "CSV: $OUT_CSV"
echo "Summary: results/summary.csv, results/summary.md"
echo "Plots: plots/val_loss.svg, plots/runtime.svg"
