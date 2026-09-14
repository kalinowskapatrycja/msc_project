import json
import tarfile
from pathlib import Path
import numpy as np

def main():
    output = Path('F:/patry/project/selection.json')
    if output.exists():
        return
    groups = {}
    for archive in sorted(Path('F:/fastmri').glob('*_multicoil_*.tar.xz')):
        parts = archive.name.removesuffix('.tar.xz').split('_')
        anatomy, split = parts[0], parts[2]
        if anatomy not in ('knee', 'brain') or split not in ('train', 'val'):
            continue
        population = 'train' if split == 'train' else 'validate'
        with tarfile.open(archive, 'r|xz') as source:
            for member in source:
                if not member.isfile() or not member.name.endswith('.h5'):
                    continue
                stem = Path(member.name).stem
                centre = 'PD' if anatomy == 'knee' else next((c for c in ('T2', 'T1', 'T1POST', 'FLAIR') if '_AX' + c + '_' in stem), None)
                if centre is None:
                    continue
                record = dict(volume=population + '_' + anatomy + '_' + stem,
                              anatomy=anatomy, centre=centre, population=population,
                              archive=str(archive), h5_member=member.name)
                groups.setdefault((population, anatomy, centre), {})[stem] = record
    training = {(anatomy, stem) for (split, anatomy, centre), pool in groups.items() if split == 'train' for stem in pool}
    rng = np.random.default_rng(20260914)
    selected = []
    for (population, anatomy, centre), pool in sorted(groups.items()):
        candidates = [record for stem, record in sorted(pool.items()) if population == 'train' or (anatomy, stem) not in training]
        count = 20 if population == 'validate' else {'PD': 350, 'T2': 900, 'T1': 80, 'T1POST': 300, 'FLAIR': 100}[centre]
        selected.extend(candidates[i] for i in rng.permutation(len(candidates))[:count])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(seed=20260914, volumes=selected), indent=2), encoding='utf-8')
    temporary.replace(output)

if __name__ == '__main__':
    main()
