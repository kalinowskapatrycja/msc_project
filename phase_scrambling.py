import numpy as np
import torch
from scipy import fft as sfft

def _sim_bloch_kernel(norm_distances, power):
    TE, TR, T1, T2, M0, num_trs = (0.005, 0.01, 1.8, 0.4, 1.0, 200)
    alpha = np.radians(7.0)
    delta_phi = np.radians(120.0)
    freq_hz = norm_distances * 200.0 * (power / 2.0)
    omega = 2 * np.pi * freq_hz
    E1_te, E2_te = (np.exp(-TE / T1), np.exp(-TE / T2))
    E1_rem, E2_rem = (np.exp(-(TR - TE) / T1), np.exp(-(TR - TE) / T2))
    phi_te = omega * TE
    cos_te, sin_te = (np.cos(phi_te), np.sin(phi_te))
    phi_rem = omega * (TR - TE)
    cos_rem, sin_rem = (np.cos(phi_rem), np.sin(phi_rem))
    cos_a, sin_a = (np.cos(alpha), np.sin(alpha))
    Mx = np.zeros_like(norm_distances, dtype=float)
    My = np.zeros_like(norm_distances, dtype=float)
    Mz = np.full_like(norm_distances, M0, dtype=float)
    current_theta = 0.0
    for tr in range(num_trs):
        current_theta = (current_theta + tr * delta_phi) % (2 * np.pi)
        cos_t, sin_t = (np.cos(current_theta), np.sin(current_theta))
        Mx1 = Mx * cos_t - My * sin_t
        My1 = Mx * sin_t + My * cos_t
        Mx2 = Mx1
        My2 = My1 * cos_a - Mz * sin_a
        Mz2 = My1 * sin_a + Mz * cos_a
        Mx_rf = Mx2 * cos_t + My2 * sin_t
        My_rf = -Mx2 * sin_t + My2 * cos_t
        Mz_rf = Mz2
        Mx_te = E2_te * (Mx_rf * cos_te - My_rf * sin_te)
        My_te = E2_te * (Mx_rf * sin_te + My_rf * cos_te)
        Mz_te = E1_te * Mz_rf + M0 * (1.0 - E1_te)
        Mx = E2_rem * (Mx_te * cos_rem - My_te * sin_rem)
        My = E2_rem * (Mx_te * sin_rem + My_te * cos_rem)
        Mz = E1_rem * Mz_te + M0 * (1.0 - E1_rem)
    mx_demod = Mx_te * np.cos(-current_theta) - My_te * np.sin(-current_theta)
    my_demod = Mx_te * np.sin(-current_theta) + My_te * np.cos(-current_theta)
    response = mx_demod + 1j * my_demod
    max_sig = np.abs(response).max()
    if max_sig > 0:
        response = response / max_sig
    return response

def sim_bloch_quadratic_bssfp_response(norm_distances, power):
    arr = np.asarray(norm_distances, dtype=float)
    if arr.size > 4096:
        uniq, inverse = np.unique(arr, return_inverse=True)
        if uniq.size * 3 < arr.size:
            return _sim_bloch_kernel(uniq, power)[inverse.ravel()].reshape(arr.shape)
    return _sim_bloch_kernel(arr, power)

def phase_tags(shape):
    rows, cols = shape
    y, x = np.ogrid[-rows / 2:rows / 2, -cols / 2:cols / 2]
    envelope = (y / (rows / 2.0)) ** 2 + (x / (cols / 2.0)) ** 2
    envelope = envelope / envelope.max()
    banding = np.angle(sim_bloch_quadratic_bssfp_response(envelope, 1.9))
    tags = np.empty((3,) + tuple(shape), dtype=np.complex64)
    tags[0] = np.exp(-1j * (np.pi * (0.6180339887 * y ** 2 + 0.4142135624 * x ** 2) + banding))
    tags[1] = 1.0
    tags[2] = np.exp(1j * (np.pi * (0.4142135624 * y ** 2 + 0.6180339887 * x ** 2) + banding))
    tags /= np.sqrt(3)
    return tags

def mixing_matrix():
    return np.exp(2j * np.pi * np.arange(3)[:, None] * np.arange(3)[None, :] / 3).astype(np.complex64)

def numpy_encoding(tags):
    p = np.fft.ifftshift(tags, axes=(-2, -1)).astype(np.complex64)[:, None]
    c = np.ones((1, 1) + tags.shape[-2:], dtype=np.complex64)
    mixing = mixing_matrix()
    def forward(u):
        images = sfft.ifft2(u[:, None], norm='ortho', workers=1)
        tagged = sfft.fft2(images * c, norm='ortho', workers=1) * p
        return np.einsum('sf,fcyx->scyx', mixing, tagged, optimize=True)
    def adjoint(y):
        back = np.einsum('sf,scyx->fcyx', mixing.conj(), y, optimize=True)
        images = sfft.ifft2(p.conj() * back, norm='ortho', workers=1)
        return sfft.fft2(np.sum(images * c.conj(), axis=1), norm='ortho', workers=1)
    return forward, adjoint

def encode_images(images, tags):
    a = np.fft.ifftshift(images, axes=(-2, -1)).astype(np.complex64)[:, None]
    c = np.ones((1, 1) + images.shape[-2:], dtype=np.complex64)
    p = np.fft.ifftshift(tags, axes=(-2, -1)).astype(np.complex64)[:, None]
    return np.einsum('sf,fcyx->scyx', mixing_matrix(), sfft.fft2(a * c, norm='ortho', workers=1) * p, optimize=True)

def torch_fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

def torch_ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

def torch_encoding(tags):
    p = torch.fft.ifftshift(tags, dim=(-2, -1)).unsqueeze(-3)
    c = torch.ones((1, 1) + tuple(tags.shape[-2:]), dtype=torch.complex64, device=tags.device)
    mixing = torch.as_tensor(mixing_matrix(), device=tags.device)
    def forward(u):
        images = torch.fft.ifft2(u.unsqueeze(-3), norm='ortho')
        tagged = torch.fft.fft2(images * c, norm='ortho') * p
        return torch.einsum('sf,...fcyx->...scyx', mixing, tagged)
    def adjoint(y):
        back = torch.einsum('sf,...scyx->...fcyx', mixing.conj(), y)
        images = torch.fft.ifft2(p.conj() * back, norm='ortho')
        return torch.fft.fft2(torch.sum(images * c.conj(), dim=-3), norm='ortho')
    return forward, adjoint
