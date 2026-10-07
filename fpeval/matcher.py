"""Keypoint-based fingerprint matchers producing similarity scores in [0, 1].

Pipeline per comparison:

1. Extract local keypoints + descriptors inside the foreground mask
   (done once per image and cached by the caller).
2. Descriptor matching with Lowe's ratio test — keep a match only if the best
   candidate is clearly better than the second best — followed by a
   one-to-one constraint (each target keypoint used at most once).
3. Geometric verification with RANSAC: fit a similarity transform
   (rotation + uniform scale + translation) and keep only matches consistent
   with it. Random descriptor coincidences between different fingers rarely
   agree on one global transform, so inlier counts separate genuine from
   impostor pairs far better than raw match counts.
4. Score = inliers / min(#keypoints_a, #keypoints_b), clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import Preprocessed

# RANSAC needs at least this many tentative matches to fit a transform.
MIN_MATCHES = 4
# Two placements of the same finger differ by at most a modest scale change.
# Rejecting anything outside this range stops RANSAC from "explaining" a
# cluster of matches with a degenerate, collapsed transform.
SCALE_RANGE = (0.75, 1.33)


@dataclass
class Features:
    keypoints: np.ndarray            # (N, 2) float32 pixel coordinates
    descriptors: np.ndarray | None   # (N, D); None if no keypoints


class KeypointMatcher:
    """Base class: subclasses only choose the detector and descriptor norm."""

    name = "base"
    norm = cv2.NORM_L2

    def __init__(self, ratio: float = 0.8, ransac_px: float = 8.0):
        self.ratio = ratio
        self.ransac_px = ransac_px
        self._bf = cv2.BFMatcher(self.norm)

    def _detector(self):
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def extract(self, pre: Preprocessed) -> Features:
        kps, desc = self._detector().detectAndCompute(pre.image, pre.mask)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
        return Features(keypoints=pts, descriptors=desc)

    def _ratio_matches(self, fa: Features, fb: Features) -> list[cv2.DMatch]:
        if fa.descriptors is None or fb.descriptors is None:
            return []
        if len(fa.descriptors) < 2 or len(fb.descriptors) < 2:
            return []
        knn = self._bf.knnMatch(fa.descriptors, fb.descriptors, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < self.ratio * n.distance]

        # Ridge texture is repetitive, so many query keypoints can pick the same
        # target keypoint. Those many-to-one matches all land on one point and
        # look like a consistent "transform" to RANSAC — keep only the best.
        best: dict[int, cv2.DMatch] = {}
        for m in good:
            if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                best[m.trainIdx] = m
        return list(best.values())

    def inliers(self, fa: Features, fb: Features) -> int:
        good = self._ratio_matches(fa, fb)
        if len(good) < MIN_MATCHES:
            return 0
        src = fa.keypoints[[m.queryIdx for m in good]]
        dst = fb.keypoints[[m.trainIdx for m in good]]
        M, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px,
            maxIters=2000, confidence=0.99,
        )
        if M is None or inlier_mask is None:
            return 0
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
            return 0
        return int(inlier_mask.sum())

    def score(self, fa: Features, fb: Features) -> float:
        denom = min(len(fa.keypoints), len(fb.keypoints))
        if denom == 0:
            return 0.0
        return float(min(self.inliers(fa, fb) / denom, 1.0))


class SiftMatcher(KeypointMatcher):
    name = "sift"
    norm = cv2.NORM_L2

    def __init__(self, n_features: int = 0, **kw):
        self.n_features = n_features
        super().__init__(**kw)

    def _detector(self):
        return cv2.SIFT_create(nfeatures=self.n_features)

    def describe(self) -> str:
        return (f"SIFT keypoints (masked, CLAHE-enhanced) + Lowe ratio test ({self.ratio}) + "
                f"RANSAC similarity-transform verification ({self.ransac_px}px); "
                f"score = inliers / min(#keypoints)")


class OrbMatcher(KeypointMatcher):
    name = "orb"
    norm = cv2.NORM_HAMMING

    def __init__(self, n_features: int = 1000, **kw):
        self.n_features = n_features
        super().__init__(**kw)

    def _detector(self):
        # Small edgeThreshold/patchSize so keypoints survive on small prints.
        return cv2.ORB_create(nfeatures=self.n_features, edgeThreshold=15, patchSize=15)

    def describe(self) -> str:
        return (f"ORB keypoints (masked, CLAHE-enhanced) + Lowe ratio test ({self.ratio}) + "
                f"RANSAC similarity-transform verification ({self.ransac_px}px); "
                f"score = inliers / min(#keypoints)")


MATCHERS = {"sift": SiftMatcher, "orb": OrbMatcher}


def get_matcher(name: str, **kw) -> KeypointMatcher:
    try:
        return MATCHERS[name](**kw)
    except KeyError:
        raise ValueError(f"unknown matcher {name!r}; choose from {sorted(MATCHERS)}") from None
