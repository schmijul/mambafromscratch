# Mamba From Scratch (CPU, C)

A local CPU-only benchmark project in plain C that compares four sequence model families under one shared pipeline:

- `MLP` baseline
- `LSTM` baseline
- `Transformer` baseline (single-head causal attention)
- `Mamba`-like selective state model

The focus is educational and reproducible local experimentation, not production optimization.

## What Is Implemented

- Dataset loader for char-level next-token prediction (`data/tinyshakespeare.txt` by default)
- Shared training and validation flow
- Manual backprop and SGD updates in C for all four models
- Common metrics: `train_loss`, `val_loss`, `val_acc`, `runtime_seconds`
- CSV benchmark logging and automatic SVG plot generation

## Build

```bash
make
```

## CLI

```bash
./bin/train --help
```

Main options:

- `--model mlp|lstm|transformer|mamba|all`
- `--data <path>`
- `--epochs <int>`
- `--steps <int>`
- `--ctx <int>`
- `--dmodel <int>`
- `--hidden <int>`
- `--lr <float>`
- `--seed <int>`
- `--benchmark <csv_path>`

## Run Benchmark + Plots

```bash
scripts/run_benchmark.sh
```

This script:

1. builds the binary,
2. runs all four models,
3. appends results to `results/benchmark.csv`,
4. generates `plots/val_loss.svg` and `plots/runtime.svg`.

## Latest Benchmark Snapshot

Configuration used for the latest rows:

- dataset: `data/tinyshakespeare.txt`
- epochs: `2`
- steps per epoch: `300`
- context length: `16`
- d_model: `16`
- hidden/state: `24`
- learning rate: `0.03`
- seed: `42`

| model | train_loss | val_loss | val_acc | seconds |
|---|---:|---:|---:|---:|
| mlp | 3.316273 | 3.078325 | 0.180328 | 0.026497 |
| lstm | 3.373384 | 3.172548 | 0.180328 | 0.147415 |
| transformer | 3.380722 | 3.195895 | 0.180328 | 0.012174 |
| mamba | 3.361045 | 3.105917 | 0.131148 | 0.037992 |

## Plots

### Validation Loss

![Validation loss](plots/val_loss.svg)

### Runtime (seconds)

![Runtime](plots/runtime.svg)

## Notes and Limitations

- The `Transformer` and `Mamba` implementations are intentionally compact v1 baselines.
- This repository prioritizes readability and local CPU comparability over scale/performance.
- For stronger conclusions, run longer schedules and multiple seeds, then compare aggregate statistics.
