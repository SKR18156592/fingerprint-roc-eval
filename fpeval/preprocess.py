"""Fingerprint image preprocessing.

Steps (each is a small, separately testable function):

1. ``load_gray``     — read image as 8-bit grayscale.
2. ``upscale``       — SOCOFing images are tiny (~96x103 px); keypoint
                       detectors need more pixels per ridge to fire reliably.
3. ``enhance``       — CLAHE to even out local contrast across the print.
4. ``foreground_mask`` — block-variance segmentation: ridges produce high
                       local variance, the flat background does not. Keypoints
                       outside the mask (border, background noise) are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Preprocessed:
    image: np.ndarray  # enhanced grayscale, uint8
    mask: np.ndarray   # uint8, 255 = fingerprint foreground


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def upscale(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def enhance(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(img)


def foreground_mask(img: np.ndarray, block: int = 16, rel_threshold: float = 0.1) -> np.ndarray:
    """Mark blocks whose intensity std-dev exceeds a fraction of the image std."""
    f = img.astype(np.float32)
    mean = cv2.blur(f, (block, block))
    sq_mean = cv2.blur(f * f, (block, block))
    local_std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))

    mask = (local_std > rel_threshold * f.std()).astype(np.uint8) * 255
    # Close small holes inside the print, then shave the noisy rim.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel)
    return mask


def preprocess(path: str, scale: float = 2.0) -> Preprocessed:
    img = upscale(load_gray(path), scale)
    img = enhance(img)
    return Preprocessed(image=img, mask=foreground_mask(img))
