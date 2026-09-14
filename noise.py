import hashlib
import numpy as np

def seed_of(*parts):
    digest = hashlib.blake2b('|'.join(repr(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), 'big')

def noise_level(seed):
    rng = np.random.default_rng(seed_of('export-noise', seed))
    if rng.random() < 0.2:
        return 0.0
    return float(1.43 * rng.random() ** 2.0)

def add_noise(encoded, seed, sigma):
    if not sigma:
        return encoded
    rng = np.random.default_rng(seed_of('noise', seed, sigma, 0))
    rms = float(np.sqrt(np.mean(np.abs(encoded) ** 2)))
    deviation = float(sigma) * rms / np.sqrt(2.0)
    noise = (rng.standard_normal(encoded.shape) + 1j * rng.standard_normal(encoded.shape)) * deviation
    return (encoded + noise).astype(encoded.dtype)
