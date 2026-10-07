"""Matcher behaviour on synthetic ridge-like images (no dataset needed)."""

import cv2
import numpy as np
import pytest

from fpeval.matcher import get_matcher
from fpeval.preprocess import Preprocessed, enhance, foreground_mask


def synthetic_print(seed: int, size: int = 200) -> np.ndarray:
    """Concentric-ish ridge pattern with random local distortions."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = rng.uniform(70, 130, 2)
    warp = cv2.GaussianBlur(rng.normal(0, 25, (size, size)).astype(np.float32), (0, 0), 12)
    r = np.hypot(x - cx, (y - cy) * rng.uniform(0.7, 1.3)) + warp
    img = 127 + 120 * np.sin(r / rng.uniform(2.5, 3.5))
    img += rng.normal(0, 25, img.shape)  # minutia-like irregularities
    return np.clip(img, 0, 255).astype(np.uint8)


def prep(img):
    img = enhance(img)
    return Preprocessed(image=img, mask=foreground_mask(img))


def rotate(img, angle, shift=(5, -4)):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += shift
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


@pytest.mark.parametrize("name", ["sift", "orb"])
def test_genuine_scores_above_impostor(name):
    m = get_matcher(name)
    a = synthetic_print(1)
    genuine = m.score(m.extract(prep(a)), m.extract(prep(rotate(a, 10))))
    impostor = m.score(m.extract(prep(a)), m.extract(prep(synthetic_print(2))))
    assert 0.0 <= impostor < genuine <= 1.0


def test_identical_images_score_high():
    m = get_matcher("sift")
    f = m.extract(prep(synthetic_print(3)))
    assert m.score(f, f) > 0.5  # i.e. more inliers than the half-score constant K


def test_empty_image_scores_zero():
    m = get_matcher("sift")
    blank = np.full((150, 150), 255, np.uint8)
    f_blank = m.extract(prep(blank))
    f_print = m.extract(prep(synthetic_print(4)))
    assert m.score(f_blank, f_print) == 0.0


def test_unknown_matcher_raises():
    with pytest.raises(ValueError):
        get_matcher("minutiae-magic")
