import numpy as np
from scipy import ndimage

def anatomy_model(anatomy):
    return {'brain': {'tissues': ('white matter', 'grey matter', 'CSF'),
               't1': (0.6, 0.95, 2.6),
               't2': (0.3, 0.45, 1.8),
               'pd': (0.75, 0.86, 1.0),
               'class_quantiles': (0.45, 0.85),
               'tissue_order': {'T2': (0, 1, 2)},
               'order_default': (2, 1, 0),
               'tuples': {'T2': ('T1', 'FLAIR'),
                          'T1': ('T2', 'T1POST'),
                          'T1POST': ('FLAIR', 'T1'),
                          'FLAIR': ('T1', 'T1POST')},
               'order': ('T1', 'T1POST', 'T2', 'FLAIR'),
               'times': {'T1': {'ti': 1.9, 'te': None, 'enhance': 0.0},
                         'T1POST': {'ti': 1.9, 'te': None, 'enhance': 0.55},
                         'T2': {'ti': None, 'te': 0.55, 'enhance': 0.0},
                         'FLAIR': {'ti': 1.8, 'te': 0.45, 'enhance': 0.0}}},
     'knee': {'tissues': ('fat/marrow', 'cartilage', 'muscle', 'fluid'),
              't1': (0.37, 1.24, 1.42, 3.0),
              't2': (0.46, 0.13, 0.11, 2.1),
              'pd': (0.9, 0.75, 0.8, 1.0),
              'class_quantiles': (0.35, 0.62, 0.82),
              'tissue_order': {'PD': (2, 1, 3, 0), 'PDFS': (0, 2, 1, 3), 'T1': (3, 2, 1, 0)},
              'order_default': (2, 1, 3, 0),
              'tuples': {'PD': ('T1', 'PDFS')},
              'order': ('T1', 'PD', 'PDFS'),
              'times': {'PD': {'ti': None,
                               'te': 0.06,
                               'tr': 2.0,
                               'enhance': 0.0,
                               'suppress': None},
                        'PDFS': {'ti': None,
                                 'te': 0.06,
                                 'tr': 2.0,
                                 'enhance': 0.0,
                                 'suppress': 'fat/marrow'},
                        'T1': {'ti': 2.2, 'te': None, 'enhance': 0.0, 'suppress': None}}}}[anatomy]

def tissue_order(source_contrast, anatomy=None):
    model = anatomy_model(anatomy)
    return model['tissue_order'].get(source_contrast, model['order_default'])

def _suppressible_tissues(model):
    return {times['suppress'] for times in model['times'].values() if times.get('suppress')}

def _proton_density(mag, level_quantile=0.4, floor=0.05):
    peak = float(mag.max())
    if peak <= 0.0:
        return np.zeros_like(mag)
    body = mag > floor * peak
    if not body.any():
        return np.zeros_like(mag)
    level = float(np.quantile(mag[body], level_quantile))
    if level <= 0.0:
        return (mag > 0.0).astype(mag.dtype)
    return np.clip(mag / level, 0.0, 1.0)

def _tissue_maps(mag, source_contrast, smooth=1.0, anatomy=None):
    model = anatomy_model(anatomy)
    t1_values, t2_values, pd_values = (model['t1'], model['t2'], model['pd'])
    suppressible = _suppressible_tissues(model)
    peak = float(mag.max())
    if peak <= 0.0:
        zeros = np.zeros(mag.shape, dtype=np.float32)
        return (zeros + 1.0, zeros + 1.0, zeros, {})
    body = mag > 0.05 * peak
    if not body.any():
        zeros = np.zeros(mag.shape, dtype=np.float32)
        return (zeros + 1.0, zeros + 1.0, zeros, {})
    cuts = [float(np.quantile(mag[body], q)) for q in model['class_quantiles']]
    cls = np.zeros(mag.shape, dtype=np.int8)
    for level, cut in enumerate(cuts):
        cls[mag > cut] = level + 1
    order = tissue_order(source_contrast, anatomy)
    t1 = np.zeros(mag.shape, dtype=np.float32)
    t2 = np.zeros(mag.shape, dtype=np.float32)
    pd = np.zeros(mag.shape, dtype=np.float32)
    membership = {}
    for level, tissue in enumerate(order):
        sel = (cls == level) & body
        t1[sel] = t1_values[tissue]
        t2[sel] = t2_values[tissue]
        pd[sel] = pd_values[tissue]
        if model['tissues'][tissue] in suppressible:
            membership[model['tissues'][tissue]] = sel.astype(np.float32)
    t1[~body] = t1_values[-1]
    t2[~body] = t2_values[-1]
    if smooth:
        t1 = ndimage.gaussian_filter(t1, smooth, mode='nearest')
        t2 = ndimage.gaussian_filter(t2, smooth, mode='nearest')
        pd = ndimage.gaussian_filter(pd, smooth, mode='nearest')
        membership = {name: ndimage.gaussian_filter(m, smooth, mode='nearest') for name, m in membership.items()}
    return (np.maximum(t1, 0.001), np.maximum(t2, 0.001), pd, membership)

def _enhancement_mask(mag, body, radius_frac=0.09):
    rows, cols = mag.shape
    idx = np.argwhere(body)
    if idx.size == 0:
        return np.zeros(mag.shape, dtype=np.float32)
    cy, cx = idx.mean(axis=0)
    cy = float(np.clip(cy + 0.12 * rows, 0, rows - 1))
    cx = float(np.clip(cx + 0.1 * cols, 0, cols - 1))
    y, x = np.ogrid[:rows, :cols]
    sigma = radius_frac * min(rows, cols)
    blob = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * sigma ** 2))
    return (blob * body).astype(np.float32)

def _contrast_weight(name, pd, t1, t2, enhancement, echo=1, anatomy=None, membership=None):
    times = anatomy_model(anatomy)['times'][name]
    ti, te, enh = (times['ti'], times['te'], times['enhance'])
    t1_eff = t1 * (1.0 - enh * enhancement) if enh else t1
    weight = pd
    suppress = times.get('suppress')
    if suppress is not None and membership:
        weight = weight * (1.0 - np.clip(membership.get(suppress, 0.0), 0.0, 1.0))
    if ti is not None:
        weight = weight * np.abs(1.0 - 2.0 * np.exp(-echo * ti / t1_eff))
    elif times.get('tr') is not None:
        weight = weight * (1.0 - np.exp(-times['tr'] / t1_eff))
    if te is not None:
        weight = weight * np.exp(-echo * te / t2)
    return weight

def render(image, anatomy, centre):
    image = image.astype(np.complex64)
    peak = np.max(np.abs(image))
    if peak > 0:
        image = image / peak
    model = anatomy_model(anatomy)
    flanks = sorted(model['tuples'][centre], key=model['order'].index)
    names = [flanks[0], centre, flanks[1]]
    mag = np.abs(image)
    t1, t2, pd_class, membership = _tissue_maps(mag, centre, anatomy=anatomy)
    pd = _proton_density(mag, 0.40) * pd_class
    enhancement = _enhancement_mask(mag, mag > 0.05 * float(mag.max()))
    result = np.empty((3,) + image.shape, dtype=np.complex64)
    for f in range(3):
        result[f] = image if f == 1 else _contrast_weight(names[f], pd, t1, t2, enhancement, anatomy=anatomy, membership=membership)
    peaks = np.max(np.abs(result), axis=(-2, -1), keepdims=True)
    targets = np.full((3, 1, 1), 0.5, dtype=np.float32)
    targets[1] = 1.0
    np.multiply(result, targets / np.where(peaks > 0, peaks, 1.0), out=result)
    return result
