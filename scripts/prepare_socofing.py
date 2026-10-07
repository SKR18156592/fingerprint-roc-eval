"""Build a ``data/subjects/person_XX/capture_K.png`` gallery from SOCOFing.

SOCOFing (Sokoto Coventry Fingerprint Dataset, Kaggle: ruizgara/socofing)
contains ONE real impression per finger (600 subjects x 10 fingers) plus
synthetically altered copies of it (Obliteration ``Obl``, Central Rotation
``CR``, Z-cut ``Zcut``) at three difficulty levels (Easy / Medium / Hard).

There are therefore no independent re-captures of the same finger. We use:

    capture_1 = Real
    capture_2..k = altered versions (see CAPTURE_SOURCES)

Because altered images are pixel-identical to the real one outside the
altered region, genuine pairs would be unrealistically easy to match. With
``--variation`` (default ON) every capture additionally gets an independent
random "acquisition" perturbation — rotation, translation, scale, blur,
sensor noise and partial occlusion — approximating what separates two real
placements of the same finger. Use ``--no-variation`` to reproduce the raw
dataset setting.

Download the dataset first, e.g.:

    curl -L -o data/raw/socofing.zip \
        https://www.kaggle.com/api/v1/datasets/download/ruizgara/socofing
    unzip -q data/raw/socofing.zip -d data/raw/socofing

then:

    python scripts/prepare_socofing.py --people 100 --captures 5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# Order in which captures are taken for each identity.
CAPTURE_SOURCES = [
    ("Real", None),
    ("Altered-Easy", "CR"),
    ("Altered-Medium", "Obl"),
    ("Altered-Hard", "Zcut"),
    ("Altered-Easy", "Obl"),
    ("Altered-Medium", "CR"),
    ("Altered-Hard", "Obl"),
]


def find_root(raw: Path) -> Path:
    """Locate the folder that contains ``Real/`` and ``Altered/``."""
    for cand in [raw, *raw.rglob("*")]:
        if (cand / "Real").is_dir() and (cand / "Altered").is_dir():
            return cand
    raise FileNotFoundError(f"no SOCOFing 'Real' + 'Altered' folders under {raw}")


def source_path(root: Path, stem: str, level: str, kind: str | None) -> Path:
    if kind is None:
        return root / "Real" / f"{stem}.BMP"
    return root / "Altered" / level / f"{stem}_{kind}.BMP"


# Every SOCOFing image carries a fixed scanner frame: 2 px grey on the
# top/left, 4 px black on the bottom/right. Left in place, its straight edges
# and corners produce identical keypoints in images of DIFFERENT people and
# inflate impostor scores, so it is cropped off before anything else.
FRAME = (2, 4)  # (top/left, bottom/right)


def strip_frame(img: np.ndarray) -> np.ndarray:
    a, b = FRAME
    return img[a:-b, a:-b]


def simulate_capture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rigid placement + imaging degradations on a white background."""
    h, w = img.shape
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-6, 6, size=2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    sigma = rng.uniform(0.0, 0.8)
    if sigma > 0.1:
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    noise = rng.normal(0, rng.uniform(2, 8), size=out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255)

    # Partial capture: blank out a band (up to 20 %) from a random side.
    band = int(rng.uniform(0, 0.2) * min(h, w))
    if band > 0:
        side = rng.integers(4)
        if side == 0:
            out[:band, :] = 255
        elif side == 1:
            out[-band:, :] = 255
        elif side == 2:
            out[:, :band] = 255
        else:
            out[:, -band:] = 255
    return out.astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw/socofing", help="extracted SOCOFing folder")
    p.add_argument("--out", default="data/subjects")
    p.add_argument("--people", type=int, default=100)
    p.add_argument("--captures", type=int, default=5, choices=range(2, len(CAPTURE_SOURCES) + 1))
    p.add_argument("--finger", default="Right_index_finger",
                   help="one finger per subject so every identity is a different person")
    p.add_argument("--variation", action=argparse.BooleanOptionalAction, default=True,
                   help="apply random acquisition variation to each capture")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = find_root(Path(args.raw))
    rng = np.random.default_rng(args.seed)
    sources = CAPTURE_SOURCES[: args.captures]

    # Subjects whose chosen finger has every required altered version.
    stems = sorted((f.stem for f in (root / "Real").glob(f"*__*_{args.finger}.BMP")),
                   key=lambda s: int(s.split("__")[0]))
    usable = [s for s in stems if all(source_path(root, s, lvl, k).exists() for lvl, k in sources)]
    if len(usable) < args.people:
        raise SystemExit(f"only {len(usable)} subjects have all {args.captures} captures")
    chosen = sorted(rng.choice(usable, size=args.people, replace=False),
                    key=lambda s: int(s.split("__")[0]))

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    width = max(2, len(str(args.people)))
    manifest = ["person,capture,source"]
    for n, stem in enumerate(chosen, 1):
        person = out / f"person_{n:0{width}d}"
        person.mkdir(parents=True)
        for c, (lvl, kind) in enumerate(sources, 1):
            src = source_path(root, stem, lvl, kind)
            img = strip_frame(cv2.imread(str(src), cv2.IMREAD_GRAYSCALE))
            if args.variation:
                img = simulate_capture(img, rng)
            cv2.imwrite(str(person / f"capture_{c}.png"), img)
            manifest.append(f"{person.name},capture_{c},{src.relative_to(root)}")

    (out / "manifest.csv").write_text("\n".join(manifest) + "\n")
    print(f"Wrote {args.people} identities x {args.captures} captures -> {out}  "
          f"(variation={'on' if args.variation else 'off'}, seed={args.seed})")


if __name__ == "__main__":
    main()
