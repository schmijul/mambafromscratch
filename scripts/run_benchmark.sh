#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results plots

make

./bin/train \
  --model all \
  --data data/tinyshakespeare.txt \
  --epochs 2 \
  --steps 300 \
  --ctx 16 \
  --dmodel 16 \
  --hidden 24 \
  --lr 0.03 \
  --seed 42 \
  --benchmark results/benchmark.csv

python3 scripts/plot_results.py results/benchmark.csv plots/val_loss.svg plots/runtime.svg

echo "Benchmark complete."
echo "CSV: results/benchmark.csv"
echo "Plots: plots/val_loss.svg, plots/runtime.svg"
