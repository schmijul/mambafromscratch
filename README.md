# Mamba From Scratch (CPU, C)

This project implements and benchmarks four sequence-model families in plain C on CPU:

- MLP baseline
- LSTM baseline
- Transformer baseline
- Mamba-like selective state model

The goal is educational: compare architectures under the same training/evaluation pipeline without external ML frameworks.

## Status

Work in progress. Core project scaffolding is in place.

## Build

```bash
make
```

## Run

```bash
./bin/train --help
```

## Planned Outputs

- `results/benchmark.csv` with comparable metrics
- `plots/` with generated charts
- README table with latest benchmark snapshot
