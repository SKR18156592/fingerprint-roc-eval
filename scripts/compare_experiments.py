"""Overlay ROC curves of several experiments and print a Markdown summary table.

Usage:
    python scripts/compare_experiments.py results/experiments/*/scores.json \
        --out docs/figures/roc_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpeval.metrics import equal_error_rate, far_resolution, roc_curve, tar_at_far  # noqa: E402


def fmt_tar(op) -> str:
    return f"{100 * op.tar:.1f}" + ("" if op.resolvable else " †")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="+")
    p.add_argument("--out", default="docs/figures/roc_comparison.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    print("| Experiment | Genuine | Impostor | EER (%) | TAR@1% | TAR@0.1% | TAR@0.01% | Min. FAR (%) |")
    print("|---|---|---|---|---|---|---|---|")
    for path in args.scores:
        name = Path(path).parent.name
        d = json.loads(Path(path).read_text())
        g, i = d["genuine"], d["impostor"]
        roc, eer = roc_curve(g, i), equal_error_rate(g, i)
        ops = [tar_at_far(g, i, f) for f in (0.01, 0.001, 0.0001)]
        print(f"| {name} | {len(g)} | {len(i)} | {100 * eer.eer:.2f} | "
              + " | ".join(fmt_tar(o) for o in ops)
              + f" | {100 * far_resolution(len(i))['smallest_nonzero']:.4f} |")
        # Draw each curve only where its impostor set can measure FAR (>= 1/n);
        # FAR = 0 points would otherwise be clipped onto the log axis and make
        # a 45-pair experiment look as if it was measured at FAR = 0.001 %.
        keep = roc.far > 0
        line, = ax.plot(roc.far[keep] * 100, roc.tar[keep] * 100, linewidth=1.8,
                        label=f"{name} (EER {100 * eer.eer:.2f}%)")
        ax.plot(roc.far[keep][-1] * 100, roc.tar[keep][-1] * 100, "o", color=line.get_color(), markersize=4)

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 100)
    ax.set_ylim(60, 100.5)
    ax.set_xlabel("FAR (%)  [log scale]")
    ax.set_ylabel("TAR (%)")
    ax.set_title("ROC comparison across experiments (dot = lowest non-zero FAR observed)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\n† = target FAR below 1/#impostor, not measurable on this set. Figure -> {args.out}")


if __name__ == "__main__":
    main()
