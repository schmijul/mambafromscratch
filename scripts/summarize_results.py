#!/usr/bin/env python3
import csv
import math
import sys
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    if len(sys.argv) != 4:
        print("Usage: summarize_results.py <benchmark.csv> <summary.csv> <summary.md>")
        sys.exit(1)

    input_csv, out_csv, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

    grouped = defaultdict(list)
    with open(input_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["model"]].append(row)

    ordered = ["mlp", "lstm", "transformer", "mamba"]
    rows = []
    for model in ordered:
        entries = grouped.get(model, [])
        if not entries:
            continue
        val_losses = [float(e["val_loss"]) for e in entries]
        val_accs = [float(e["val_acc"]) for e in entries]
        runtimes = [float(e["seconds"]) for e in entries]
        params = [int(e["params"]) for e in entries]
        targets = [int(e["target_params"]) for e in entries]
        d_model = [int(e["d_model"]) for e in entries]
        hidden = [int(e["hidden"]) for e in entries]

        rows.append(
            {
                "model": model,
                "runs": len(entries),
                "mean_val_loss": mean(val_losses),
                "std_val_loss": std(val_losses),
                "mean_val_acc": mean(val_accs),
                "mean_seconds": mean(runtimes),
                "params": int(round(mean(params))),
                "target_params": int(round(mean(targets))),
                "d_model": int(round(mean(d_model))),
                "hidden": int(round(mean(hidden))),
            }
        )

    rows.sort(key=lambda r: r["mean_val_loss"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "runs",
                "d_model",
                "hidden",
                "target_params",
                "params",
                "mean_val_loss",
                "std_val_loss",
                "mean_val_acc",
                "mean_seconds",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["model"],
                    r["runs"],
                    r["d_model"],
                    r["hidden"],
                    r["target_params"],
                    r["params"],
                    f"{r['mean_val_loss']:.6f}",
                    f"{r['std_val_loss']:.6f}",
                    f"{r['mean_val_acc']:.6f}",
                    f"{r['mean_seconds']:.6f}",
                ]
            )

    winner = rows[0]
    runner_up = rows[1] if len(rows) > 1 else rows[0]
    margin = runner_up["mean_val_loss"] - winner["mean_val_loss"]

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Aggregated Results (lower val_loss is better)\n\n")
        f.write(f"Winner: **{winner['model']}**\n")
        f.write(f"Margin vs runner-up: **{margin:.6f}** val_loss points\n\n")
        f.write("| model | runs | d_model | hidden | target_params | params | mean_val_loss | std_val_loss | mean_val_acc | mean_seconds |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['runs']} | {r['d_model']} | {r['hidden']} | {r['target_params']} | {r['params']} | {r['mean_val_loss']:.6f} | {r['std_val_loss']:.6f} | {r['mean_val_acc']:.6f} | {r['mean_seconds']:.6f} |\n"
            )

    print(f"winner={winner['model']} margin={margin:.6f}")


if __name__ == "__main__":
    main()
