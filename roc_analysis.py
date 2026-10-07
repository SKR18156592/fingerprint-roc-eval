"""ROC analysis of genuine / impostor scores.

Usage:
    python roc_analysis.py --scores results/scores.json --out-dir results

Produces:
    score_distribution.png   genuine (green) vs impostor (red) histogram + EER line
    roc_curve.png            TAR vs FAR with EER point and TAR @ FAR=1% / 0.1%
    summary.json             every number printed below, machine-readable
and prints the EER, TAR @ fixed FAR, FAR resolution and a threshold
sensitivity table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fpeval.metrics import equal_error_rate, far_frr_at, far_resolution, roc_curve, tar_at_far
from fpeval.plots import plot_roc, plot_score_distribution

TARGET_FARS = (0.01, 0.001, 0.0001)            # 1 %, 0.1 %, 0.01 %
DEFAULT_TABLE = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
BASELINE_THRESHOLD = 0.40                      # hand-picked threshold this analysis replaces


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="results/scores.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_TABLE),
                   help="thresholds for the sensitivity table")
    return p.parse_args()


def pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def row_note(t: float, far: float, frr: float) -> str:
    notes = []
    if abs(t - BASELINE_THRESHOLD) < 1e-9:
        notes.append("(hand-picked baseline)")
    if far > 0.10:
        notes.append("Too permissive")
    elif frr > 0.50:
        notes.append("Too restrictive")
    return " ".join(notes)


def threshold_table(genuine, impostor, thresholds, eer) -> list[dict]:
    rows = []
    for t in thresholds:
        far, frr = far_frr_at(genuine, impostor, t)
        rows.append({"label": f"{t:.2f}", "threshold": t, "tar": 1 - frr, "far": far, "frr": frr,
                     "note": row_note(t, far, frr)})
    far, frr = far_frr_at(genuine, impostor, eer.threshold)
    rows.append({"label": "EER pt", "threshold": eer.threshold, "tar": 1 - frr, "far": far, "frr": frr,
                 "note": f"<- EER (t = {eer.threshold:.3f})"})
    return sorted(rows, key=lambda r: r["threshold"])


def print_table(rows) -> None:
    print("\nThreshold | TAR (%) | FAR (%) | FRR (%) | Notes")
    print("----------|---------|---------|---------|------------------")
    for r in rows:
        print(f"  {r['label']:<7} |  {pct(r['tar'])}  |  {pct(r['far'])}  |  {pct(r['frr'])}  | {r['note']}")


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.scores).read_text())
    genuine = np.asarray(data["genuine"])
    impostor = np.asarray(data["impostor"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roc = roc_curve(genuine, impostor)
    eer = equal_error_rate(genuine, impostor)
    ops = [tar_at_far(genuine, impostor, f) for f in TARGET_FARS]
    res = far_resolution(impostor.size)

    print("=== ROC Analysis Results ===")
    print(f"Matcher:              {data.get('metadata', {}).get('matcher_used', 'n/a')}")
    print(f"Genuine pairs:        {genuine.size}")
    print(f"Impostor pairs:       {impostor.size}")
    print(f"EER:                  {100 * eer.eer:.2f}%   (threshold = {eer.threshold:.3f})")
    for op in ops:
        flag = "" if op.resolvable else "   [NOT resolvable: target < 1/#impostor]"
        print(f"TAR @ FAR = {100 * op.target_far:g}%:".ljust(22)
              + f"{100 * op.tar:.1f}%   (threshold = {op.threshold:.3f}){flag}")
    print(f"Smallest measurable FAR: {100 * res['smallest_nonzero']:.4f}%  (1/{impostor.size})")
    print(f"Reliable FAR floor:      {100 * res['reliable_floor']:.4f}%  (~10 errors needed)")
    print(f"Recommended threshold: {eer.threshold:.3f}  (at EER)")

    rows = threshold_table(genuine, impostor, args.thresholds, eer)
    print_table(rows)

    plot_score_distribution(genuine, impostor, eer, str(out_dir / "score_distribution.png"))
    plot_roc(roc, eer, ops[:2], str(out_dir / "roc_curve.png"))

    summary = {
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "eer": eer.eer,
        "eer_threshold": eer.threshold,
        "operating_points": [op.__dict__ for op in ops],
        "far_resolution": res,
        "genuine_stats": {"mean": float(genuine.mean()), "std": float(genuine.std()),
                          "min": float(genuine.min()), "max": float(genuine.max())},
        "impostor_stats": {"mean": float(impostor.mean()), "std": float(impostor.std()),
                           "min": float(impostor.min()), "max": float(impostor.max())},
        "overlap": {"genuine_below_max_impostor": float(np.mean(genuine <= impostor.max())),
                    "impostor_above_min_genuine": float(np.mean(impostor >= genuine.min()))},
        "threshold_table": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved plots + summary.json -> {out_dir}/")


if __name__ == "__main__":
    main()
