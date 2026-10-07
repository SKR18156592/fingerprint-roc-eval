"""Dataset loading.

Expected on-disk layout (one folder per identity, several captures each):

    data/subjects/
    ├── person_01/
    │   ├── capture_1.png
    │   ├── capture_2.png
    │   └── capture_3.png
    ├── person_02/
    ...

``scripts/prepare_socofing.py`` builds this layout from the SOCOFing dataset;
any other source (e.g. phone-camera captures) works if it follows the same
structure.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gallery(root: str | Path, min_captures: int = 2) -> dict[str, list[Path]]:
    """Return ``{person_id: [capture paths, sorted]}``.

    Identities with fewer than ``min_captures`` images are skipped, since they
    cannot form a genuine pair.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data folder not found: {root}")

    gallery: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        captures = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if len(captures) >= min_captures:
            gallery[person_dir.name] = captures

    if len(gallery) < 2:
        raise ValueError(f"need at least 2 identities with >= {min_captures} captures in {root}")
    return gallery


def balance(gallery: dict[str, list[Path]], n_captures: int | None = None) -> dict[str, list[Path]]:
    """Truncate every identity to the same number of captures.

    Equal capture counts keep the closed-form pair formulas exact and stop one
    identity from dominating the genuine distribution.
    """
    k = n_captures or min(len(v) for v in gallery.values())
    return {pid: caps[:k] for pid, caps in gallery.items() if len(caps) >= k}
