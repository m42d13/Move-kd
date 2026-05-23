#!/usr/bin/env python
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def moving_average(values, window):
    if window <= 1:
        return values
    smoothed = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def load_scalars(logdir):
    event_files = sorted(Path(logdir).glob("*/events.out.tfevents.*"), key=os.path.getmtime)
    if not event_files:
        event_files = sorted(Path(logdir).glob("events.out.tfevents.*"), key=os.path.getmtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    scalars = defaultdict(dict)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                scalars[tag][event.step] = {
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "tag": tag,
                    "value": float(event.value),
                    "event_file": str(event_file),
                }
    rows = []
    for tag in sorted(scalars):
        rows.extend(scalars[tag][step] for step in sorted(scalars[tag]))
    return event_files, rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "wall_time", "tag", "value", "event_file"],
        )
        writer.writeheader()
        writer.writerows(rows)


def latest_by_tag(rows):
    latest = {}
    for row in rows:
        tag = row["tag"]
        if tag not in latest or row["step"] >= latest[tag]["step"]:
            latest[tag] = row
    return latest


def plot(rows, path, smooth_window, total_steps):
    by_tag = defaultdict(list)
    for row in rows:
        by_tag[row["tag"]].append(row)
    for tag in by_tag:
        by_tag[tag].sort(key=lambda row: row["step"])

    loss_tags = [tag for tag in sorted(by_tag) if "loss" in tag.lower()]
    lr_tags = [tag for tag in sorted(by_tag) if "learning_rate" in tag.lower() or tag.endswith("/lr")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for tag in loss_tags:
        xs = [row["step"] for row in by_tag[tag]]
        ys = [row["value"] for row in by_tag[tag]]
        axes[0].plot(xs, moving_average(ys, smooth_window), label=f"{tag} ma{smooth_window}", linewidth=1.8)
        if tag.endswith("/loss") or tag == "train/loss":
            axes[0].plot(xs, ys, alpha=0.25, linewidth=0.8, label=f"{tag} raw")

    axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss components")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    for tag in lr_tags:
        xs = [row["step"] for row in by_tag[tag]]
        ys = [row["value"] for row in by_tag[tag]]
        axes[1].plot(xs, ys, label=tag, linewidth=1.8)

    axes[1].set_ylabel("learning rate")
    axes[1].set_xlabel("optimizer step")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    if total_steps:
        axes[1].set_xlim(left=0, right=total_steps)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MoVE-KD fine-tune TensorBoard metrics.")
    parser.add_argument(
        "--logdir",
        default="checkpoints/finetune/full-adapter-move-kd-7b-v1.1-metrics/runs",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints/finetune/full-adapter-move-kd-7b-v1.1-metrics/plots",
    )
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=83162)
    args = parser.parse_args()

    event_files, rows = load_scalars(args.logdir)
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "training_metrics.csv"
    json_path = out_dir / "latest_metrics.json"
    png_path = out_dir / "training_metrics.png"

    write_csv(rows, csv_path)
    latest = latest_by_tag(rows)
    with json_path.open("w") as f:
        json.dump(
            {
                "event_files": [str(path) for path in event_files],
                "latest": latest,
            },
            f,
            indent=2,
        )
    plot(rows, png_path, args.smooth, args.total_steps)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    for tag, row in sorted(latest.items()):
        print(f"{tag}: step={row['step']} value={row['value']:.6g}")


if __name__ == "__main__":
    main()
