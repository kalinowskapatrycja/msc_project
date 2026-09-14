import torch

def fftc(x, dims):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)

def ifftc(x, dims):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)

def drop_readout_oversampling(kspace):
    hyb = ifftc(kspace, dims=(-2,))
    n = hyb.shape[-2]
    keep = n // 2
    start = (n - keep) // 2
    return fftc(hyb[..., start:start + keep, :], dims=(-2,))

def acs_centre(mask):
    m = mask.to(torch.bool)
    lo = hi = m.numel() // 2
    while lo - 1 >= 0 and bool(m[lo - 1]):
        lo -= 1
    while hi + 1 < m.numel() and bool(m[hi + 1]):
        hi += 1
    return (lo + hi) // 2

def acs_width(mask):
    m = mask.to(torch.bool)
    lo = hi = m.numel() // 2
    while lo - 1 >= 0 and bool(m[lo - 1]):
        lo -= 1
    while hi + 1 < m.numel() and bool(m[hi + 1]):
        hi += 1
    return hi - lo + 1

def calib_matrix(cal, k):
    nc = cal.shape[0]
    win = cal.unfold(1, k, 1).unfold(2, k, 1)
    nwx, nwy = (win.shape[1], win.shape[2])
    A = win.permute(1, 2, 0, 3, 4).reshape(nwx * nwy, nc * k * k)
    return A.contiguous().to(torch.complex64)

def espirit_maps(kspace, k, calib, cal_centre, sigma2_cut, eig_thresh, nmaps):
    nc, nx, ny = kspace.shape
    device = kspace.device
    rx, ry = calib
    cx, cy = cal_centre
    x0, y0 = (cx - rx // 2, cy - ry // 2)
    cal = kspace[:, x0:x0 + rx, y0:y0 + ry]
    A = calib_matrix(cal, k)
    _, svals, vh = torch.linalg.svd(A.cpu(), full_matrices=False)
    svals, vh = (svals.to(device), vh.to(device))
    n_ker = int(torch.sum(svals ** 2 >= sigma2_cut * svals[0] ** 2))
    kernels = vh[:n_ker].conj().reshape(n_ker, nc, k, k)
    GG = torch.zeros((nx * ny, nc, nc), dtype=torch.complex64, device=device)
    sx, sy = (nx // 2 - k // 2, ny // 2 - k // 2)
    for lo in range(0, n_ker, 32):
        hi = min(lo + 32, n_ker)
        pad = torch.zeros((hi - lo, nc, nx, ny), dtype=torch.complex64, device=device)
        pad[:, :, sx:sx + k, sy:sy + k] = torch.flip(kernels[lo:hi], dims=(-2, -1)).conj()
        gimg = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pad, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)) / k
        G = gimg.reshape(hi - lo, nc, nx * ny).permute(2, 1, 0).contiguous()
        GG += G @ G.conj().mT
    maps = torch.zeros((nmaps, nx * ny, nc), dtype=torch.complex64, device=device)
    eigs = torch.zeros((nmaps, nx * ny), dtype=torch.float32, device=device)
    chunk = 2048
    lo = 0
    while lo < nx * ny:
        hi = min(lo + chunk, nx * ny)
        try:
            w, v = torch.linalg.eigh(GG[lo:hi])
        except torch.OutOfMemoryError:
            if chunk <= 128:
                raise
            torch.cuda.empty_cache()
            chunk //= 2
            continue
        for j in range(nmaps):
            maps[j, lo:hi] = v[:, :, -1 - j]
            eigs[j, lo:hi] = w[:, -1 - j].to(torch.float32)
        lo = hi
    maps = maps.permute(0, 2, 1).reshape(nmaps, nc, nx, ny)
    eigs = eigs.reshape(nmaps, nx, ny)
    inside = eigs[0] >= eig_thresh
    ref = int(torch.argmax(maps[0][:, inside].abs().sum(dim=1)))
    maps = maps * torch.exp(-1j * torch.angle(maps[:, ref:ref + 1]))
    for j in range(nmaps):
        maps[j][:, eigs[j] < eig_thresh] = 0
    return maps, eigs

def bcast_mask(mask, shape):
    m = mask.reshape(1, -1).expand(shape[-2], shape[-1])
    return m.to(torch.float32).unsqueeze(0)

def make_operators(maps, mask):
    m2 = bcast_mask(mask, maps.shape)

    def fwd(m):
        coils = torch.einsum('jcxy,jxy->cxy', maps, m)
        return fftc(coils, dims=(-2, -1)) * m2

    def adj(y):
        coils = ifftc(y * m2, dims=(-2, -1))
        return torch.einsum('jcxy,cxy->jxy', maps.conj(), coils)
    return (fwd, adj)

def cg_solve(apply_A, b, iters, rtol=1e-12):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    rs0 = rs
    for _ in range(iters):
        if not torch.isfinite(rs) or rs <= rtol * rs0:
            break
        Ap = apply_A(p)
        pAp = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
        if not torch.isfinite(pAp) or pAp <= 0:
            break
        a = rs / pAp
        x = x + (a * p).to(x.dtype)
        r = r - (a * Ap).to(r.dtype)
        rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        p = r + rs_new / rs * p
        rs = rs_new
    return x

def soft_sense(ksp, maps, mask, alpha, iters):
    fwd, adj = make_operators(maps, mask)
    y = ksp * bcast_mask(mask, ksp.shape)
    return cg_solve(lambda m: adj(fwd(m)) + alpha * m, adj(y), iters)

def combine_slice(kspace, mask, calib=24, nmaps=2, cg_iters=50):
    kspace = drop_readout_oversampling(kspace)
    width = acs_width(mask)
    eff_calib = max(6 + 1, min(calib, width))
    maps, eigs = espirit_maps(kspace, k=6, calib=(eff_calib, eff_calib), cal_centre=(kspace.shape[-2] // 2, acs_centre(mask)), sigma2_cut=0.001, eig_thresh=0.9, nmaps=nmaps)
    m = soft_sense(kspace, maps, mask, alpha=0.001, iters=cg_iters)
    return m[0], float((eigs[0] >= 0.9).float().mean())
