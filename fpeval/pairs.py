"""Genuine / impostor pair generation.

Given ``{person_id: [capture_0, capture_1, ...]}`` this builds the comparison
lists used for score collection.

Genuine pairs  — every unordered pair of captures of the SAME person:
    N_genuine = N_people * C(N_captures, 2) = N_people * k(k-1)/2

Impostor pairs — two protocols are supported:

* ``"reference"`` — one reference capture per person, compared across all
  pairs of people:  N_impostor = C(N_people, 2) = n(n-1)/2.
  Cheap, and every person contributes equally, but it throws away most of the
  cross-person comparisons that the data already contains.

* ``"cross"`` (default) — every capture of person A against every capture of
  person B:  N_impostor = C(N_people, 2) * k^2.
  Uses all available impostor evidence, which matters because the smallest
  measurable FAR is 1 / N_impostor.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Hashable, Mapping, Sequence

IMPOSTOR_MODES = ("cross", "reference")

Pair = tuple[Hashable, Hashable]


def genuine_pairs(gallery: Mapping[Hashable, Sequence[Hashable]]) -> list[Pair]:
    pairs: list[Pair] = []
    for captures in gallery.values():
        pairs.extend(combinations(captures, 2))
    return pairs


def impostor_pairs(gallery: Mapping[Hashable, Sequence[Hashable]], mode: str = "cross") -> list[Pair]:
    if mode not in IMPOSTOR_MODES:
        raise ValueError(f"mode must be one of {IMPOSTOR_MODES}, got {mode!r}")

    people = list(gallery)
    pairs: list[Pair] = []
    for a, b in combinations(people, 2):
        if mode == "reference":
            pairs.append((gallery[a][0], gallery[b][0]))
        else:
            pairs.extend(product(gallery[a], gallery[b]))
    return pairs


def expected_counts(n_people: int, n_captures: int, mode: str = "cross") -> dict:
    """Closed-form pair counts, used to sanity-check the generated lists."""
    n_gen = n_people * n_captures * (n_captures - 1) // 2
    person_pairs = n_people * (n_people - 1) // 2
    n_imp = person_pairs if mode == "reference" else person_pairs * n_captures**2
    return {"genuine": n_gen, "impostor": n_imp}
