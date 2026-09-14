import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim._functional import adam
from threadpoolctl import threadpool_limits
from phase_scrambling import phase_tags, torch_encoding, torch_fft2c, torch_ifft2c
from noise import seed_of

def initialize_weights():
    torch.manual_seed(seed_of('learned-init', 0) % (2 ** 31))
    weights = {}
    for cascade in range(12):
        for layer, incoming, outgoing in [('body.0', 11, 96), ('body.2', 96, 96), ('body.4', 96, 96), ('out', 96, 6)]:
            weight = torch.empty((outgoing, incoming, 3, 3))
            bias = torch.empty(outgoing)
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(incoming * 9)
            torch.nn.init.uniform_(bias, -bound, bound)
            if layer == 'out':
                weight.zero_()
                bias.zero_()
            weights[f'blocks.{cascade}.{layer}.weight'] = weight.to('cuda').requires_grad_()
            weights[f'blocks.{cascade}.{layer}.bias'] = bias.to('cuda').requires_grad_()
    return dict(step_size=torch.zeros(12, device='cuda', requires_grad=True), **weights)

def reconstruct(weights, batch):
    forward, adjoint = torch_encoding(batch['tags'])
    mask = torch.fft.ifftshift(batch['mask'].to(torch.complex64), dim=(-2, -1)).unsqueeze(-3)
    state = torch_fft2c(batch['recon_all'])
    zf = torch.view_as_real(batch['zf_centre']).movedim(-1, 1)
    for cascade in range(12):
        images = torch_ifft2c(state)
        channels = torch.cat((torch.view_as_real(images).movedim(-1, 2).flatten(1, 2), zf, batch['mask'].to(images.real.dtype)), dim=1)
        for layer in ['body.0', 'body.2', 'body.4', 'out']:
            prefix = f'blocks.{cascade}.{layer}'
            channels = torch.nn.functional.conv2d(channels, weights[prefix + '.weight'], weights[prefix + '.bias'], padding=1)
            if layer != 'out':
                channels = torch.nn.functional.relu(channels, inplace=True)
        residual = torch.view_as_complex(channels.float().unflatten(1, (-1, 2)).movedim(2, -1).contiguous())
        state = state + torch_fft2c(residual)
        normal = torch.fft.fftshift(adjoint(mask * forward(torch.fft.ifftshift(state, dim=(-2, -1)))), dim=(-2, -1))
        rhs = torch.fft.fftshift(adjoint(torch.fft.ifftshift(batch['kspace'], dim=(-2, -1))), dim=(-2, -1))
        state = state - weights['step_size'][cascade] / 1.0 * (normal - rhs)
    return torch_ifft2c(state)

def load_batch(indices, samples, tags_cache):
    batch = {k: [] for k in ('recon_all', 'truth_all', 'zf_centre', 'kspace', 'mask', 'scale')}
    for index in indices:
        with np.load(Path('F:/patry/project/samples') / f'{index:05d}.npz') as sample:
            scale = float(np.percentile(np.abs(sample['zf_centre']), 99.5))
            for key in ('recon_all', 'truth_all', 'zf_centre'):
                batch[key].append(torch.from_numpy(sample[key]).to(torch.complex64) / scale)
            mask = sample['mask']
            kspace = np.zeros(mask.shape, dtype=np.complex64)
            kspace[mask] = sample['kspace_samples']
            batch['kspace'].append(torch.from_numpy(kspace[:, None]) / scale)
            batch['mask'].append(torch.from_numpy(mask))
            batch['scale'].append(torch.tensor(scale, dtype=torch.float32))
    batch = {k: torch.stack(v).to('cuda') for k, v in batch.items()}
    shape = tuple(batch['mask'].shape[-2:])
    if shape not in tags_cache:
        tags_cache[shape] = torch.as_tensor(phase_tags(shape), device='cuda')
    batch['tags'] = tags_cache[shape]
    batch['entries'] = [samples[index] for index in indices]
    return batch

def source_batches(indices, samples, epoch, shuffle):
    groups = {}
    for index in indices:
        groups.setdefault(tuple(samples[index]['matrix']), []).append(index)
    batches = []
    for shape, members in sorted(groups.items()):
        if shuffle:
            rng = np.random.default_rng(seed_of('learned-batches', 'train', '2d', 0, 0, epoch, shape))
            members = [members[i] for i in rng.permutation(len(members))]
        batches.extend([members[i:i + 4] for i in range(0, len(members), 4)])
    if shuffle:
        rng = np.random.default_rng(seed_of('learned-batch-order', 'train', '2d', 0, 0, epoch))
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches

def balanced_batches(indices, samples, steps):
    groups = {}
    for index in indices:
        entry = samples[index]
        groups.setdefault((entry['anatomy'], entry['centre']), {}).setdefault(tuple(entry['matrix']), []).append(index)
    keys = sorted(groups)
    weights = np.array([1. if k[0] == 'knee' else 1. / sum(x[0] == 'brain' for x in keys) for k in keys])
    weights /= weights.sum()
    rng = np.random.default_rng(20260911)
    queues = {k: [] for k in keys}
    for _ in range(steps):
        key = keys[int(rng.choice(len(keys), p=weights))]
        if not queues[key]:
            chunks = []
            for shape, members in sorted(groups[key].items()):
                shuffled = rng.permutation(members).tolist()
                size = min(4, max(1, 600000 // (shape[-1] * shape[-2])))
                chunks.extend([shuffled[i:i + size] for i in range(0, len(shuffled), size)])
            rng.shuffle(chunks)
            queues[key] = chunks
        yield queues[key].pop()

def ssim_kernel(dtype, device):
    x = torch.arange(-5, 6, dtype=torch.float64)
    kernel = torch.exp(-(x ** 2) / (2.0 * 1.5 ** 2))
    return (kernel / kernel.sum()).to(dtype=dtype, device=device)

def _smooth(x, kernel):
    radius = (kernel.numel() - 1) // 2
    x = x.unsqueeze(1)
    x = torch.nn.functional.pad(x, (0, 0, radius, radius), mode='replicate')
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, -1, 1))
    x = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode='replicate')
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, 1, -1))
    return x.squeeze(1)

def data_range(image, quantile=0.995):
    flat = image.abs().flatten(1).to(torch.float64)
    value = torch.quantile(flat, quantile, dim=1)
    return torch.where(value > 0, value, torch.ones_like(value)).detach()

def ssim(truth, pred, drange=None):
    a, b = (truth.abs().to(torch.float64), pred.abs().to(torch.float64))
    dr = data_range(truth) if drange is None else drange.to(torch.float64)
    dr = dr.reshape(-1, *[1] * (a.dim() - 1))
    c1, c2 = ((0.01 * dr) ** 2, (0.03 * dr) ** 2)
    kernel = ssim_kernel(a.dtype, a.device)
    mu_a, mu_b = (_smooth(a, kernel), _smooth(b, kernel))
    var_a = _smooth(a * a, kernel) - mu_a * mu_a
    var_b = _smooth(b * b, kernel) - mu_b * mu_b
    cov = _smooth(a * b, kernel) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return (num / den).flatten(1).mean(dim=1)

def pathway_loss(truth, pred, ssim_weight):
    l1 = (pred - truth).abs().flatten(1).mean(dim=1)
    return l1 + ssim_weight * (1.0 - ssim(truth, pred).to(l1.dtype))

def source_loss(prediction, truth):
    centre = pathway_loss(truth[:, 1], prediction[:, 1], 0.01).mean()
    flanks = torch.stack([pathway_loss(truth[:, f], prediction[:, f], 0.01).mean() for f in (0, 2)]).mean()
    return centre + 0.1 * flanks

def pathway_rmse(prediction, batch):
    error = prediction - batch['truth_all']
    return torch.linalg.vector_norm(error.flatten(2), dim=2) / math.sqrt(error.shape[-1] * error.shape[-2]) * batch['scale'][:, None]

def continuation_loss(prediction, batch, reference):
    errors = pathway_rmse(prediction, batch)
    scales = [[reference[e['anatomy'] + '/' + e['centre']]['rmse'][p]['mean'] for p in ('left', 'centre', 'right')] for e in batch['entries']]
    scales = torch.tensor(scales, dtype=errors.dtype, device='cuda').clamp_min(1e-8)
    weights = torch.full((3,), 1.0, device='cuda')
    return ((errors / scales) * weights).sum(1).mean() / weights.sum()

def evaluate(weights, indices, samples, tags_cache):
    groups = {}
    with torch.no_grad():
        for members in source_batches(indices, samples, 0, False):
            batch = load_batch(members, samples, tags_cache)
            errors = pathway_rmse(reconstruct(weights, batch), batch).cpu().numpy()
            for entry, row in zip(batch['entries'], errors):
                groups.setdefault(entry['anatomy'] + '/' + entry['centre'], []).append(row.astype(np.float64))
    return {key: {'rmse': {p: {'mean': float(np.mean(np.array(rows)[:, i])), 'median': float(np.median(np.array(rows)[:, i]))} for i, p in enumerate(('left', 'centre', 'right'))}} for key, rows in sorted(groups.items())}

def selection(metrics, reference):
    anatomy = {}
    for key, result in metrics.items():
        ratios = [result['rmse'][p][s] / max(reference[key]['rmse'][p][s], 1e-12) for p in ('left', 'centre', 'right') for s in ('mean', 'median')]
        anatomy.setdefault(key.split('/')[0], []).extend(ratios)
    return float(np.mean([np.mean(values) for values in anatomy.values()]))

def update_average(average, weights, decay):
    with torch.no_grad():
        for name, value in weights.items():
            average[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

def adam_state(weights):
    return dict(mean=[torch.zeros_like(v) for v in weights.values()],
                square=[torch.zeros_like(v) for v in weights.values()],
                steps=[torch.tensor(0.0) for v in weights.values()])

def adam_step(weights, optimizer, rate):
    with torch.no_grad():
        adam(list(weights.values()), [v.grad for v in weights.values()],
             optimizer['mean'], optimizer['square'], [], optimizer['steps'],
             foreach=True, amsgrad=False, beta1=0.9, beta2=0.999,
             lr=rate, weight_decay=0.0, eps=1e-8, maximize=False)

def restore_training(payload):
    weights = {k: v.detach().requires_grad_() for k, v in payload['model'].items()}
    optimizer = payload['optimiser']
    optimizer['steps'] = [v.cpu() for v in optimizer['steps']]
    return weights, optimizer, payload['ema'], payload['step']

def save_checkpoint(path, weights, optimizer, average, step, best):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model={k: v.detach() for k, v in weights.items()}, optimiser=optimizer, ema=average, step=step, best_score=best, architecture=dict(nF=3, num_shots=3, cascades=12, channels=96, convs=3)), temporary)
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))

def train_source(population, tags_cache):
    directory = Path('F:/patry/project/training/source')
    directory.mkdir(parents=True, exist_ok=True)
    weights = initialize_weights()
    optimizer = adam_state(weights)
    average = {k: v.detach().clone() for k, v in weights.items()}
    start = 0
    best = float('inf')
    if (directory / 'latest.pt').exists():
        payload = torch.load(directory / 'latest.pt', weights_only=False, map_location='cuda')
        weights, optimizer, average, start = restore_training(payload)
        best = payload['best_score']
    for epoch in range(start, 250):
        factor = (epoch + 1) / 12.0 if epoch < 12 else 0.5 * (1.0 + math.cos(math.pi * min(1.0, (epoch - 12) / 238)))
        for indices in source_batches(population['source_train'], population['samples'], epoch, True):
            batch = load_batch(indices, population['samples'], tags_cache)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = source_loss(reconstruct(weights, batch), batch['truth_all'])
            for value in weights.values():
                value.grad = None
            loss.backward()
            adam_step(weights, optimizer, 0.0002 * factor)
            update_average(average, weights, 0.999)
        metrics = evaluate(average, population['source_validation'], population['samples'], tags_cache)
        score = metrics['knee/PD']['rmse']['centre']['mean']
        if score < best:
            best = score
            save_checkpoint(directory / 'selected.pt', average, optimizer, average, epoch + 1, best)
        save_checkpoint(directory / 'latest.pt', weights, optimizer, average, epoch + 1, best)
    return torch.load(directory / 'selected.pt', weights_only=False, map_location='cuda')['model']

def train_continuation(index, budget, rate, decay, parent, population, reference, tags_cache):
    directory = Path('F:/patry/project/training') / f'joint-{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().clone().requires_grad_() for k, v in parent.items()}
    optimizer = adam_state(weights)
    average = {k: v.detach().clone() for k, v in weights.items()}
    start = 0
    if (directory / 'latest.pt').exists():
        payload = torch.load(directory / 'latest.pt', weights_only=False, map_location='cuda')
        weights, optimizer, average, start = restore_training(payload)
        best = payload['best_score']
    else:
        metrics = evaluate(weights, population['validation'], population['samples'], tags_cache)
        best = selection(metrics, reference)
        save_checkpoint(directory / 'selected.pt', weights, optimizer, average, 0, best)
        save_checkpoint(directory / 'latest.pt', weights, optimizer, average, 0, best)
    for step, indices in enumerate(balanced_batches(population['train'], population['samples'], budget)):
        if step < start:
            continue
        batch = load_batch(indices, population['samples'], tags_cache)
        warmup = min(100, max(1, budget // 20))
        progress = max(0.0, (step - warmup) / max(1, budget - warmup))
        factor = min(1.0, (step + 1) / warmup) * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
        for value in weights.values():
            value.grad = None
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = continuation_loss(reconstruct(weights, batch), batch, reference)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(weights.values()), 1.0, error_if_nonfinite=True)
        adam_step(weights, optimizer, rate * factor)
        update_average(average, weights, decay)
        if (step + 1) % (800 if index == 0 else 2000) == 0 or step + 1 == budget:
            metrics = evaluate(average, population['validation'], population['samples'], tags_cache)
            score = selection(metrics, reference)
            if score < best:
                best = score
                save_checkpoint(directory / 'selected.pt', average, optimizer, average, step + 1, best)
        if (step + 1) % 100 == 0 or step + 1 == budget:
            save_checkpoint(directory / 'latest.pt', weights, optimizer, average, step + 1, best)
    return torch.load(directory / 'selected.pt', weights_only=False, map_location='cuda')['model']

def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    population = json.loads(Path('F:/patry/project/population.json').read_text())
    tags_cache = {}
    with threadpool_limits(limits=2):
        weights = train_source(population, tags_cache)
        reference = evaluate(weights, population['validation'], population['samples'], tags_cache)
        Path('F:/patry/project/training/source/baseline.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
        stages = [(1600, 0.0001, 0.995), (16000, 0.0001, 0.998),
                  (16000, 0.0001, 0.998), (16000, 0.0001, 0.998),
                  (32000, 0.0001, 0.998), (32000, 0.00006, 0.998),
                  (32000, 0.00006, 0.998), (32000, 0.00006, 0.998),
                  (32000, 0.00006, 0.998), (32000, 0.00003, 0.998),
                  (6000, 0.00003, 0.998)]
        for index, (budget, rate, decay) in enumerate(stages):
            weights = train_continuation(index, budget, rate, decay, weights, population, reference, tags_cache)
        metrics = evaluate(weights, population['validation'], population['samples'], tags_cache)
    output = Path('F:/patry/project/training')
    result = dict(model=weights, architecture=dict(nF=3, num_shots=3, cascades=12, channels=96, convs=3), metrics=metrics, selection_score=selection(metrics, reference))
    torch.save(result, output / 'retrained.pt')
    reloaded = torch.load(output / 'retrained.pt', weights_only=False, map_location='cuda')
    with threadpool_limits(limits=2):
        metrics = evaluate(reloaded['model'], population['validation'], population['samples'], tags_cache)
    (output / 'metrics.json').write_text(json.dumps(dict(metrics=metrics, score=selection(metrics, reference)), indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
