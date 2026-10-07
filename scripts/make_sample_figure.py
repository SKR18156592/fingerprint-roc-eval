"""Save a figure of raw SOCOFing images vs. the simulated captures built from them.

Usage:
    python scripts/make_sample_figure.py --people 2 --out docs/figures/sample_captures.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/subjects")
    p.add_argument("--raw", default="data/raw/socofing/SOCOFing")
    p.add_argument("--people", type=int, default=2)
    p.add_argument("--out", default="docs/figures/sample_captures.png")
    args = p.parse_args()

    data = Path(args.data)
    rows = list(csv.DictReader(open(data / "manifest.csv")))
    people = sorted({r["person"] for r in rows})[: args.people]
    n_caps = max(int(r["capture"].split("_")[1]) for r in rows)

    fig, axes = plt.subplots(2 * len(people), n_caps, figsize=(1.6 * n_caps, 3.6 * len(people)))
    for pi, person in enumerate(people):
        for r in (r for r in rows if r["person"] == person):
            c = int(r["capture"].split("_")[1]) - 1
            src = cv2.imread(str(Path(args.raw) / r["source"]), cv2.IMREAD_GRAYSCALE)
            sim = cv2.imread(str(data / person / f"{r['capture']}.png"), cv2.IMREAD_GRAYSCALE)
            label = Path(r["source"]).stem.split("finger")[-1].strip("_") or "Real"
            level = Path(r["source"]).parent.name.replace("Altered-", "")
            for row, img, title in [(2 * pi, src, f"{level} {label}" if label != "Real" else "Real"),
                                    (2 * pi + 1, sim, "simulated capture")]:
                ax = axes[row, c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title, fontsize=7)
                ax.axis("off")
        axes[2 * pi, 0].text(-0.15, 0.5, person, transform=axes[2 * pi, 0].transAxes,
                             rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("SOCOFing source images (top) and simulated captures used for evaluation (bottom)", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
