import numpy as np
from scipy.special import j1

def _weighted_without_replacement(weights, count, rng):
    weights = np.asarray(weights, dtype=np.float64).ravel()
    count = int(min(count, np.count_nonzero(weights > 0)))
    if count <= 0:
        return np.empty(0, dtype=np.intp)
    keys = rng.random(weights.size)
    with np.errstate(divide='ignore'):
        np.log(keys, out=keys)
        keys /= -np.maximum(weights, 1e-30)
    if count >= weights.size:
        return np.arange(weights.size)[weights > 0]
    return np.argpartition(keys, count - 1)[:count]

def shot_masks(shape, family, seed):
    rows, cols = shape
    y, x = np.ogrid[-rows / 2:rows / 2, -cols / 2:cols / 2]
    distances = np.hypot(x, y)
    max_dist = distances.max()
    radii, inverse = np.unique(distances / max_dist, return_inverse=True)
    if family == 'Exponential':
        profile = np.exp(-6.0 * radii)
    elif family == 'Polynomial':
        profile = (1.0 - radii) ** 6.0
    else:
        scaled = radii * 1.0 * np.pi * 5
        scaled = np.where(scaled == 0, 1e-5, scaled)
        profile = np.abs(j1(scaled) / scaled)
        peak = float(profile.max())
        profile = profile / peak if peak > 0 else profile
    centre = distances <= max_dist * 0.06
    probabilities = np.ascontiguousarray(profile[inverse.ravel()].reshape(rows, cols), dtype=np.float32)
    probabilities[centre] = 0.0
    needed = int(rows * cols / 12.0) - int(centre.sum())
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(3):
        mask = centre.copy()
        if needed > 0:
            mask.ravel()[_weighted_without_replacement(probabilities.ravel(), needed, rng)] = True
        masks.append(mask)
    return np.stack(masks)

def undersample(noisy, mask):
    return np.fft.fftshift(noisy * np.fft.ifftshift(mask, axes=(-2, -1))[:, None], axes=(-2, -1))
