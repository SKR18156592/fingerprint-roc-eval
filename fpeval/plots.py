"""Plotting helpers for score distributions and ROC curves."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from .metrics import EerResult, OperatingPoint, RocCurve

GENUINE_COLOR = "#2e9e44"
IMPOSTOR_COLOR = "#d43d3d"
EER_COLOR = "#222222"


def plot_score_distribution(genuine, impostor, eer: EerResult, out_path: str, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.linspace(0.0, 1.0, bins + 1)

    ax.hist(impostor, bins=edges, color=IMPOSTOR_COLOR, alpha=0.55, label=f"Impostor (n={len(impostor)})")
    ax.hist(genuine, bins=edges, color=GENUINE_COLOR, alpha=0.55, label=f"Genuine (n={len(genuine)})")
    ax.axvline(eer.threshold, color=EER_COLOR, linestyle="--", linewidth=1.5,
               label=f"EER threshold = {eer.threshold:.3f}")

    # Impostor counts usually dwarf genuine counts; a log axis keeps both visible.
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Genuine vs Impostor Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(roc: RocCurve, eer: EerResult, operating_points: list[OperatingPoint], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    far_pct, tar_pct = roc.far * 100, roc.tar * 100

    ax.plot(far_pct, tar_pct, color="#1f5fbf", linewidth=2, label="ROC")

    eer_pct = eer.eer * 100
    ax.plot(eer_pct, 100 - eer_pct, "o", color=EER_COLOR, markersize=8,
            label=f"EER = {eer_pct:.2f}% (t = {eer.threshold:.3f})")

    styles = ["#e08a00", "#8a2be2", "#008b8b"]
    for op, color in zip(operating_points, styles):
        note = "" if op.resolvable else " — below FAR resolution"
        ax.axhline(op.tar * 100, color=color, linestyle=":", linewidth=1.5,
                   label=f"TAR @ FAR={op.target_far * 100:g}% = {op.tar * 100:.1f}%{note}")
        ax.axvline(op.target_far * 100, color=color, linestyle=":", linewidth=0.8, alpha=0.6)

    # Log-scale FAR is the standard way to read the low-FAR region that
    # matters for deployment; clip at the smallest strictly positive FAR.
    positive = far_pct[far_pct > 0]
    ax.set_xscale("log")
    ax.set_xlim(max(positive.min() / 2, 1e-3) if positive.size else 1e-3, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("False Accept Rate — FAR (%)  [log scale]")
    ax.set_ylabel("True Accept Rate — TAR (%)")
    ax.set_title("ROC Curve — Fingerprint Matching")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
