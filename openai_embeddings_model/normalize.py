import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    """Convert (n_samples, dim) embedding to unit vector"""
    norm: np.ndarray = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0  # Avoid zero division by setting zero vectors to 1.0
    return vec / norm
