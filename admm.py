import numpy as np
import torch
from phase_scrambling import torch_encoding

def hankel_geometry(shape):
    ny, nx = shape
    y, x = np.ogrid[-2:3, -2:3]
    offsets = np.argwhere(x ** 2 + y ** 2 <= 4) - [2, 2]
    slices = [(slice(2 + dy, ny - 2 + dy), slice(2 + dx, nx - 2 + dx)) for dy, dx in offsets]
    counts = np.zeros(shape, dtype=np.float32)
    for sy, sx in slices:
        counts[sy, sx] += 1.0
    return slices, np.where(counts == 0.0, 1.0, counts)

def gpu_cg(apply, b, x):
    x = x.clone()
    r = b - apply(x)
    p = r.clone()
    rs = torch.vdot(r.flatten(), r.flatten()).real
    zero = torch.zeros((), dtype=rs.dtype, device=rs.device)
    for _ in range(6):
        Ap = apply(p)
        denominator = torch.vdot(p.flatten(), Ap.flatten()).real
        alpha = torch.where(denominator > 0, rs / denominator, zero)
        x = x + alpha * p
        r = r - alpha * Ap
        new = torch.vdot(r.flatten(), r.flatten()).real
        p = r + torch.where(rs > 0, new / rs, zero) * p
        rs = new
    return x

def gpu_prox(K, slices, weights):
    ny, nx = K.shape[-2:]
    lifted = torch.stack([K[:, sy, sx] for sy, sx in slices], dim=1).reshape(39, (ny - 4) * (nx - 4))
    double = lifted.to(torch.complex128)
    gram = double @ double.conj().T
    _, vectors = np.linalg.eigh(gram.cpu().numpy().astype(np.complex128))
    basis = torch.as_tensor(vectors[:, -31:].astype(np.complex64), device=K.device)
    patches = (basis @ (basis.conj().T @ lifted)).reshape(3, 13, ny - 4, nx - 4)
    result = torch.zeros_like(K)
    for j, (sy, sx) in enumerate(slices):
        result[:, sy, sx] += patches[:, j]
    return result / weights

def reconstruct_gpu(y, mask, tags):
    with torch.inference_mode():
        tags = torch.as_tensor(tags, device='cuda')
        forward, adjoint = torch_encoding(tags)
        m = torch.fft.ifftshift(torch.as_tensor(mask, device='cuda'), dim=(-2, -1)).to(torch.complex64)[:, None]
        rhs = adjoint(m * torch.fft.ifftshift(torch.as_tensor(y, device='cuda'), dim=(-2, -1)))
        K, Z, U = rhs.clone(), rhs.clone(), torch.zeros_like(rhs)
        slices, counts = hankel_geometry(y.shape[-2:])
        weights = torch.as_tensor(counts, device='cuda')
        def apply(x):
            return adjoint(m * forward(x)) + 0.2 * x
        for _ in range(60):
            previous = Z
            K = gpu_cg(apply, rhs + 0.2 * (Z - U), K)
            Z = torch.fft.ifftshift(gpu_prox(torch.fft.fftshift(K + U, dim=(-2, -1)), slices, weights), dim=(-2, -1))
            U = U + (K - Z)
            primal = torch.linalg.vector_norm(K - Z)
            dual = 0.2 * torch.linalg.vector_norm(Z - previous)
            scale = torch.clamp(torch.maximum(torch.linalg.vector_norm(K), torch.linalg.vector_norm(Z)), min=1e-12)
            if bool((primal / scale < 1e-4) & (dual / scale < 1e-4)):
                break
        return torch.fft.fftshift(torch.fft.ifft2(K, norm='ortho'), dim=(-2, -1)).cpu().numpy()
