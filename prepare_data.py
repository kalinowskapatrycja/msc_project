import json
import shutil
import tarfile
from pathlib import Path
import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from espirit import combine_slice
from noise import seed_of

def combine_volume(path, destination, record):
    with h5py.File(path, 'r') as h, torch.inference_mode():
        kspace = h['kspace']
        mask = torch.ones(kspace.shape[-1], dtype=torch.float32, device='cuda')
        combined = {}
        for index in range(1, kspace.shape[0] - 1):
            k = torch.from_numpy(np.asarray(kspace[index]).astype(np.complex64)).to('cuda')
            image, inside = combine_slice(k, mask)
            if inside >= 0.5:
                combined[str(index)] = image.cpu().numpy()
        if record['population'] == 'validate':
            rng = np.random.default_rng(seed_of('validation-slices', record['volume'], 20260914))
            keys = rng.permutation(list(combined))[:4]
            combined = {key: combined[key] for key in sorted(keys, key=int)}
        temporary = destination.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **combined)
        temporary.replace(destination)

def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    selection = json.loads(Path('F:/patry/project/selection.json').read_text())
    output = Path('F:/patry/project/combined')
    output.mkdir(parents=True, exist_ok=True)
    archives = {}
    for record in selection['volumes']:
        if not (output / (record['volume'] + '.npz')).exists():
            archives.setdefault(record['archive'], {})[record['h5_member']] = record
    with threadpool_limits(limits=2):
        for archive, wanted in archives.items():
            with tarfile.open(archive, 'r|xz') as source:
                for member in source:
                    if member.name not in wanted:
                        continue
                    record = wanted.pop(member.name)
                    volume = record['volume']
                    temporary = output / (volume + '.h5')
                    with source.extractfile(member) as incoming, temporary.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
                    combine_volume(temporary, output / (volume + '.npz'), record)
                    temporary.unlink()
                    if not wanted:
                        break
    write_population(selection, output)

def write_population(selection, output):
    samples = []
    for record in selection['volumes']:
        with np.load(output / (record['volume'] + '.npz')) as combined:
            for key in sorted(combined.files, key=int):
                matrix = [3, *combined[key].shape]
                for family in ('Bessel', 'Exponential', 'Polynomial'):
                    samples.append(dict(volume=record['volume'], anatomy=record['anatomy'],
                                        centre=record['centre'], population=record['population'],
                                        slice=int(key), family=family, matrix=matrix))
    population = dict(samples=samples,
                      train=[i for i, e in enumerate(samples) if e['population'] == 'train'],
                      validation=[i for i, e in enumerate(samples) if e['population'] == 'validate'])
    rng = np.random.default_rng(20260914)
    for split, count in (('train', 40), ('validate', 10)):
        volumes = sorted({e['volume'] for e in samples if e['population'] == split and e['anatomy'] == 'knee'})
        selected = {}
        for volume in rng.permutation(volumes)[:count]:
            slices = sorted({e['slice'] for e in samples if e['volume'] == volume})
            selected[volume] = set(rng.permutation(slices)[:4])
        population['source_train' if split == 'train' else 'source_validation'] = [i for i, e in enumerate(samples) if e['volume'] in selected and e['slice'] in selected[e['volume']]]
    temporary = output.parent / 'population.tmp'
    temporary.write_text(json.dumps(population, indent=2), encoding='utf-8')
    temporary.replace(output.parent / 'population.json')

if __name__ == '__main__':
    main()
