"""Fundus image preprocessing with OpenCV.

Steps:
1. Crop the dark border around the circular fundus.
2. Resize to a square of `size` pixels.
3. Ben Graham's local-colour normalisation (subtract a Gaussian blur) which
   evens out lighting and makes lesions (microaneurysms, exudates) stand out.
4. Mask everything outside the fundus circle.
"""
import cv2
import numpy as np


def crop_dark_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():  # completely dark image: leave it untouched
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def ben_graham(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def circle_mask(img: np.ndarray, scale: float = 0.95) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, -1)
    return img * mask[..., None]


def preprocess_array(img_rgb: np.ndarray, size: int = 224, enhance: bool = True) -> np.ndarray:
    """RGB uint8 array -> preprocessed RGB uint8 array of shape (size, size, 3)."""
    img = crop_dark_border(img_rgb)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if enhance:
        img = ben_graham(img)
    return circle_mask(img)


def load_image(path, size: int = 224, enhance: bool = True) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return preprocess_array(rgb, size, enhance)
