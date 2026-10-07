"""Biometric verification metrics computed from genuine / impostor score sets.

Convention used throughout: scores are similarities in [0, 1] (higher = more
alike) and a comparison is ACCEPTED when ``score >= threshold``.

    FAR(t) = #impostor scores >= t / #impostor      (false accepts)
    FRR(t) = #genuine  scores <  t / #genuine       (false rejects)
    TAR(t) = 1 - FRR(t)

All rates are returned as fractions (0..1); callers convert to % for display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_array(scores) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("score list is empty")
    return arr


def far_frr_at(genuine, impostor, threshold: float) -> tuple[float, float]:
    """Return (FAR, FRR) at a single threshold."""
    g, i = _as_array(genuine), _as_array(impostor)
    far = float(np.mean(i >= threshold))
    frr = float(np.mean(g < threshold))
    return far, frr


@dataclass
class RocCurve:
    thresholds: np.ndarray  # ascending
    far: np.ndarray         # non-increasing as threshold grows
    frr: np.ndarray         # non-decreasing as threshold grows

    @property
    def tar(self) -> np.ndarray:
        return 1.0 - self.frr


def roc_curve(genuine, impostor) -> RocCurve:
    """Evaluate FAR/FRR at every distinct score (plus both extremes).

    Every change in FAR or FRR happens exactly at an observed score, so using
    the observed scores as candidate thresholds gives the exact empirical
    curve — no fixed-step sweep that could skip over a crossing point.
    """
    g = np.sort(_as_array(genuine))
    i = np.sort(_as_array(impostor))
    thr = np.unique(np.concatenate([g, i, [0.0, np.nextafter(1.0, 2.0)]]))

    # searchsorted(side="left") = count of values strictly below threshold
    frr = np.searchsorted(g, thr, side="left") / g.size
    far = 1.0 - np.searchsorted(i, thr, side="left") / i.size
    return RocCurve(thresholds=thr, far=far, frr=frr)


@dataclass
class EerResult:
    eer: float        # equal error rate (fraction)
    threshold: float  # threshold at which FAR ~= FRR


def equal_error_rate(genuine, impostor) -> EerResult:
    """Find the point where FAR and FRR cross.

    FAR - FRR is monotonically non-increasing in the threshold. We locate the
    first threshold where it becomes <= 0 and linearly interpolate between that
    threshold and the previous one, which gives a stable estimate even when
    the empirical curves are step functions.
    """
    roc = roc_curve(genuine, impostor)
    diff = roc.far - roc.frr
    k = int(np.argmax(diff <= 0))  # first index with FAR <= FRR

    if k == 0:
        return EerResult(eer=float((roc.far[0] + roc.frr[0]) / 2), threshold=float(roc.thresholds[0]))

    d0, d1 = diff[k - 1], diff[k]
    w = d0 / (d0 - d1) if d0 != d1 else 0.5  # fraction of the way from k-1 to k
    far = roc.far[k - 1] + w * (roc.far[k] - roc.far[k - 1])
    frr = roc.frr[k - 1] + w * (roc.frr[k] - roc.frr[k - 1])
    thr = roc.thresholds[k - 1] + w * (roc.thresholds[k] - roc.thresholds[k - 1])
    return EerResult(eer=float((far + frr) / 2), threshold=float(min(thr, 1.0)))


@dataclass
class OperatingPoint:
    target_far: float
    tar: float
    far: float          # FAR actually achieved (<= target)
    threshold: float
    resolvable: bool    # False if target is below the dataset's FAR resolution


def tar_at_far(genuine, impostor, target_far: float) -> OperatingPoint:
    """Highest TAR whose FAR does not exceed ``target_far``.

    If ``target_far`` is smaller than 1 / #impostor the dataset cannot tell
    FAR = target apart from FAR = 0, so the point is flagged unresolvable.
    """
    i = _as_array(impostor)
    roc = roc_curve(genuine, impostor)
    ok = roc.far <= target_far + 1e-12
    # Lowest threshold meeting the FAR constraint maximises TAR.
    k = int(np.argmax(ok))
    return OperatingPoint(
        target_far=target_far,
        tar=float(roc.tar[k]),
        far=float(roc.far[k]),
        threshold=float(min(roc.thresholds[k], 1.0)),
        resolvable=target_far >= 1.0 / i.size,
    )


def far_resolution(n_impostor: int) -> dict:
    """How small a FAR the impostor set can actually support.

    * ``smallest_nonzero`` — one false accept out of n (1/n).
    * ``rule_of_three_95`` — with ZERO observed false accepts, the 95 % upper
      confidence bound on the true FAR is ~3/n.
    * ``reliable_floor`` — ~10 observed errors are needed for a usable
      estimate, so FAR values below 10/n are not reliably measured.
    """
    if n_impostor <= 0:
        raise ValueError("n_impostor must be positive")
    return {
        "smallest_nonzero": 1.0 / n_impostor,
        "rule_of_three_95": 3.0 / n_impostor,
        "reliable_floor": 10.0 / n_impostor,
    }
