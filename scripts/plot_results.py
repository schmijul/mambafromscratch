#!/usr/bin/env python3
import csv
import os
import sys
from collections import defaultdict


def aggregate(path):
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["model"]].append(row)

    ordered = ["mlp", "lstm", "transformer", "mamba"]
    labels = []
    val_losses = []
    runtimes = []
    for model in ordered:
        rows = grouped.get(model, [])
        if not rows:
            continue
        labels.append(model)
        val_losses.append(sum(float(r["val_loss"]) for r in rows) / len(rows))
        runtimes.append(sum(float(r["seconds"]) for r in rows) / len(rows))
    return labels, val_losses, runtimes


def draw_bar_svg(path, title, ylabel, labels, values, color):
    width = 920
    height = 440
    margin_left = 84
    margin_right = 30
    margin_top = 60
    margin_bottom = 92
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    bar_count = len(values)
    slot_w = chart_w / max(1, bar_count)
    bar_w = slot_w * 0.62

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="#f6f8fb"/>')
    lines.append(f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#1f2a37">{title}</text>')

    x0, y0 = margin_left, margin_top
    x1, y1 = margin_left + chart_w, margin_top + chart_h
    lines.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#4b5563" stroke-width="1.4"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#4b5563" stroke-width="1.4"/>')

    for i in range(5):
        frac = i / 4.0
        y = y1 - frac * chart_h
        val = frac * max_v
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{x0-10}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#374151">{val:.3f}</text>')

    for i, (label, val) in enumerate(zip(labels, values)):
        cx = x0 + (i + 0.5) * slot_w
        h = (val / max_v) * chart_h
        bx = cx - bar_w / 2
        by = y1 - h
        lines.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" rx="6"/>')
        lines.append(f'<text x="{cx:.2f}" y="{y1+24}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">{label}</text>')
        lines.append(f'<text x="{cx:.2f}" y="{by-8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#111827">{val:.4f}</text>')

    lines.append(f'<text x="20" y="{height/2:.2f}" transform="rotate(-90 20,{height/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">{ylabel}</text>')
    lines.append('</svg>')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    if len(sys.argv) != 4:
        print("Usage: plot_results.py <benchmark.csv> <val_loss.svg> <runtime.svg>")
        sys.exit(1)

    csv_path, val_plot, runtime_plot = sys.argv[1], sys.argv[2], sys.argv[3]
    labels, val_losses, runtimes = aggregate(csv_path)

    os.makedirs(os.path.dirname(val_plot), exist_ok=True)
    os.makedirs(os.path.dirname(runtime_plot), exist_ok=True)

    draw_bar_svg(val_plot, "Mean Validation Loss by Model", "mean val_loss", labels, val_losses, "#2563eb")
    draw_bar_svg(runtime_plot, "Mean Runtime by Model (seconds)", "mean seconds", labels, runtimes, "#059669")


if __name__ == "__main__":
    main()
