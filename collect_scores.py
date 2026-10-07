"""Collect genuine and impostor match scores and save them to JSON.

Usage:
    python collect_scores.py --data data/subjects --out results/scores.json
    python collect_scores.py --matcher orb --impostor-mode reference

Steps:
    1. Load ``data/<person>/<capture>`` images.
    2. Preprocess + extract features ONCE per image (cached).
    3. Build all genuine pairs and all impostor pairs.
    4. Score every pair, save scores + metadata to ``scores.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fpeval.dataset import balance, load_gallery
from fpeval.matcher import MATCHERS, get_matcher
from fpeval.pairs import IMPOSTOR_MODES, expected_counts, genuine_pairs, impostor_pairs
from fpeval.preprocess import preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/subjects", help="folder with one sub-folder per identity")
    p.add_argument("--out", default="results/scores.json")
    p.add_argument("--matcher", choices=sorted(MATCHERS), default="sift")
    p.add_argument("--impostor-mode", choices=IMPOSTOR_MODES, default="cross",
                   help="cross: all captures vs all captures; reference: first capture per person only")
    p.add_argument("--captures", type=int, default=None, help="captures per identity (default: min available)")
    p.add_argument("--scale", type=float, default=2.0, help="upscale factor before feature extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gallery = balance(load_gallery(args.data), args.captures)
    n_people = len(gallery)
    n_captures = len(next(iter(gallery.values())))
    matcher = get_matcher(args.matcher)

    print(f"Identities: {n_people}   captures/identity: {n_captures}   matcher: {matcher.name}")

    # 1) Feature extraction, once per image.
    t0 = time.perf_counter()
    features = {}
    for caps in gallery.values():
        for path in caps:
            features[path] = matcher.extract(preprocess(path, scale=args.scale))
    t_extract = time.perf_counter() - t0
    print(f"Extracted features for {len(features)} images in {t_extract:.1f}s")

    # 2) Pairs.
    gen_pairs = genuine_pairs(gallery)
    imp_pairs = impostor_pairs(gallery, args.impostor_mode)
    expected = expected_counts(n_people, n_captures, args.impostor_mode)
    assert len(gen_pairs) == expected["genuine"] and len(imp_pairs) == expected["impostor"]
    print(f"Genuine pairs: {len(gen_pairs)}   Impostor pairs: {len(imp_pairs)}")

    # 3) Scoring.
    t0 = time.perf_counter()
    genuine = [matcher.score(features[a], features[b]) for a, b in gen_pairs]
    impostor = []
    for idx, (a, b) in enumerate(imp_pairs, 1):
        impostor.append(matcher.score(features[a], features[b]))
        if idx % 5000 == 0:
            print(f"  scored {idx}/{len(imp_pairs)} impostor pairs")
    t_match = time.perf_counter() - t0
    n_comparisons = len(gen_pairs) + len(imp_pairs)
    print(f"Scored {n_comparisons} comparisons in {t_match:.1f}s "
          f"({1000 * t_match / n_comparisons:.2f} ms/comparison)")

    k = n_captures
    result = {
        "genuine": [round(s, 6) for s in genuine],
        "impostor": [round(s, 6) for s in impostor],
        "metadata": {
            "n_people": n_people,
            "n_captures_per_person": n_captures,
            "n_genuine_pairs": len(gen_pairs),
            "n_impostor_pairs": len(imp_pairs),
            "genuine_formula": f"{n_people} x ({k} x {k - 1} / 2) = {len(gen_pairs)}",
            "impostor_formula": (
                f"{n_people} x {n_people - 1} / 2 = {len(imp_pairs)}" if args.impostor_mode == "reference"
                else f"({n_people} x {n_people - 1} / 2) x {k}^2 = {len(imp_pairs)}"
            ),
            "impostor_mode": args.impostor_mode,
            "matcher_used": matcher.describe(),
            "upscale_factor": args.scale,
            "data_root": str(args.data),
            "identities": list(gallery),
            "timing_sec": {"feature_extraction": round(t_extract, 2), "matching": round(t_match, 2)},
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
