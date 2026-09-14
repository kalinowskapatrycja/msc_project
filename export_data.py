import json
from pathlib import Path
import numpy as np
import torch
from scipy import fft as sfft
from threadpoolctl import threadpool_limits
from contrasts import render
from phase_scrambling import phase_tags, encode_images, numpy_encoding
from noise import seed_of, noise_level, add_noise
from undersampling import shot_masks, undersample
from admm import reconstruct_gpu

def export_sample(entry, image, destination):
    truth = render(image, entry['anatomy'], entry['centre'])
    tags = phase_tags(truth.shape[-2:])
    seed = seed_of(entry['volume'], entry['slice'], 4.0, entry['family'], 3, 0.0)
    sigma = noise_level(seed)
    encoded = encode_images(truth, tags)
    noisy = add_noise(encoded, seed, sigma)
    mask = shot_masks(truth.shape[-2:], entry['family'], seed)
    kspace = undersample(noisy, mask)
    recon = reconstruct_gpu(kspace, mask, tags)
    _, adjoint = numpy_encoding(tags)
    zero_filled = np.fft.fftshift(sfft.ifft2(adjoint(np.fft.ifftshift(kspace, axes=(-2, -1))), norm='ortho', workers=1), axes=(-2, -1))[1]
    denominator = float(np.vdot(zero_filled, zero_filled).real)
    zero_filled = zero_filled * (np.vdot(zero_filled, truth[1]) / denominator if denominator > 0 else 1.0)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, recon_all=recon, truth_all=truth, zf_centre=zero_filled, kspace_samples=kspace[:, 0][mask], mask=mask)
    temporary.replace(destination)

def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = False
    population = json.loads(Path('F:/patry/project/population.json').read_text())
    output = Path('F:/patry/project/samples')
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for index, entry in enumerate(population['samples']):
        if not (output / f'{index:05d}.npz').exists():
            groups.setdefault(entry['volume'], []).append((index, entry))
    with threadpool_limits(limits=1):
        for volume, entries in groups.items():
            with np.load(Path('F:/patry/project/combined') / (volume + '.npz')) as data:
                combined = {key: data[key] for key in data.files}
            for index, entry in entries:
                export_sample(entry, combined[str(entry['slice'])], output / f'{index:05d}.npz')

if __name__ == '__main__':
    main()
