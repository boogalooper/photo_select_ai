from __future__ import annotations

import cv2
import numpy as np


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def technical_quality(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean()) / 255.0
    std = float(gray.std()) / 64.0
    exposure = 1.0 - min(1.0, abs(mean - 0.50) / 0.50)
    clipped_black = float(np.mean(gray <= 3))
    clipped_white = float(np.mean(gray >= 252))
    clipping = clamp01(1.0 - (clipped_black + clipped_white) * 3.0)
    contrast = clamp01(std)
    return clamp01(0.45 * exposure + 0.30 * clipping + 0.25 * contrast)


def sharpness_score(rgb: np.ndarray) -> float:
    if rgb.size == 0 or min(rgb.shape[:2]) < 3:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Soft saturation; robust across resized previews.
    return clamp01(variance / (variance + 180.0))


def appearance_descriptor(rgb: np.ndarray) -> list[float]:
    if rgb.size == 0:
        return []
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-8)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray)[:4, :4].flatten()
    dct = dct / (np.linalg.norm(dct) + 1e-8)
    return np.concatenate([hist, dct]).astype(float).tolist()
