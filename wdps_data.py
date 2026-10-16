"""Multi-sensor registry, leave-one-sensor-out folds, degradation simulator and fixed test sets."""
import numpy as np, json, os, zlib

CUBES = os.environ.get('WDPS_CUBES', './cubes')
META = json.load(open(f'{CUBES}/meta.json'))

# Leave-one-sensor-family-out folds. AVIRIS (train scenes), APEX, SAMSON and FENIX are always in the training pool.
FOLDS = {
    'ROSIS':    ['paviaU', 'paviaC'],
    'HYDICE':   ['urban', 'dcmall'],
    'Hyperion': ['botswana'],
    'VNIR-CASI/AISA': ['muufl64', 'houston13', 'houston18', 'trento'],
    'Natural':  ['cave_peppers', 'cave_pompoms'] + [f'kaist{i:02d}' for i in range(1, 11)],
}
ALWAYS_TRAIN = ['jasper', 'sandiego', 'saltonsea', 'apex', 'samson', 'fenix']
XSCENE = ['salinas', 'cuprite']           # AVIRIS scenes never used for training (cross-scene protocol)
REAL = ['indianpines220', 'salinasA224', 'enmap', 'muufl72']

_cache = {}
def load(name):
    if name not in _cache:
        d = np.load(f'{CUBES}/{name}.npz', allow_pickle=True)
        _cache[name] = {k: d[k] for k in d.files}
    return _cache[name]

def train_pool(fold):
    held = set(FOLDS[fold])
    pool = list(ALWAYS_TRAIN)
    for f, names in FOLDS.items():
        if f != fold: pool += names
    return [p for p in pool if p not in held]

# ------------------------------------------------------------------ degradations
def _sigma_bands(rng, B, lo=10 / 255, hi=70 / 255):
    return rng.uniform(lo, hi, B).astype(np.float32)

def degrade(x, rng, kinds, params=None):
    """x: (H,W,B) clean in [0,1]. kinds subset of {'gauss','stripe','deadline','impulse','poisson','cstripe'}.
    Returns noisy y and the parameter dict (re-usable to generate a view with identical degradation statistics)."""
    H, W, B = x.shape
    p = dict(params or {})
    y = x.copy()
    if 'poisson' in kinds:          # signal-dependent Poisson-Gaussian (unseen during training)
        a = p.setdefault('peak', float(rng.uniform(20, 60)))
        y = rng.poisson(np.clip(y, 0, None) * a).astype(np.float32) / a
    if 'gauss' in kinds:
        s = p.setdefault('sig', _sigma_bands(rng, B) if 'sig_scale' not in p else _sigma_bands(rng, B) * p['sig_scale'])
        if len(s) != B: s = np.interp(np.linspace(0, 1, B), np.linspace(0, 1, len(s)), s).astype(np.float32)
        y = y + rng.standard_normal(y.shape).astype(np.float32) * s[None, None, :]
    if 'stripe' in kinds:
        frac_b = p.setdefault('stripe_bands', 1 / 3); frac_c = p.setdefault('stripe_cols', float(rng.uniform(0.05, 0.15)))
        bands = rng.choice(B, max(1, int(B * frac_b)), replace=False)
        for b in bands:
            cols = rng.choice(W, max(1, int(W * frac_c)), replace=False)
            y[:, cols, b] += rng.uniform(-0.25, 0.25, len(cols)).astype(np.float32)[None, :]
    if 'cstripe' in kinds:          # spectrally coherent stripes: the same columns over contiguous band blocks (unseen)
        nblk = p.setdefault('cstripe_blocks', 3)
        for _ in range(nblk):
            b0 = rng.integers(0, max(1, B - B // 6)); bl = np.arange(b0, min(B, b0 + max(2, B // 6)))
            cols = rng.choice(W, max(1, int(W * 0.08)), replace=False); off = rng.uniform(-0.2, 0.2, len(cols)).astype(np.float32)
            y[:, cols[:, None], bl[None, :]] += off[None, :, None]
    if 'deadline' in kinds:
        bands = rng.choice(B, max(1, int(B * p.setdefault('dead_bands', 1 / 3))), replace=False)
        for b in bands:
            nc = rng.integers(max(1, W // 64), max(2, W // 12) + 1)
            cols = rng.choice(W, nc, replace=False)
            for c in cols:
                wdt = rng.integers(1, 4); y[:, c:c + wdt, b] = 0.0
    if 'impulse' in kinds:
        bands = rng.choice(B, max(1, int(B * p.setdefault('imp_bands', 1 / 3))), replace=False)
        for b in bands:
            r = rng.uniform(0.1, 0.3); m = rng.random((H, W)) < r
            y[:, :, b][m] = (rng.random(m.sum()) > 0.5).astype(np.float32)
    return y.astype(np.float32), p

TEST_CASES = {
    'C1 Non-i.i.d. Gaussian': ['gauss'],
    'C2 Gaussian+Stripe': ['gauss', 'stripe'],
    'C3 Gaussian+Deadline': ['gauss', 'deadline'],
    'C4 Gaussian+Impulse': ['gauss', 'impulse'],
    'C5 Mixture': ['gauss', 'stripe', 'deadline', 'impulse'],
    'C6 Unseen (Poisson+coherent stripe)': ['poisson', 'gauss', 'cstripe'],
}
TRAIN_KINDS = ['stripe', 'deadline', 'impulse']   # Gaussian always present; each extra kind with p=0.5

def sample_train_kinds(rng):
    return ['gauss'] + [k for k in TRAIN_KINDS if rng.random() < 0.5]

# ------------------------------------------------------------------ crops
def valid_crop_origins(x, size, n, rng, min_valid=0.97):
    H, W, _ = x.shape; out = []; tries = 0
    m = x.mean(2) > 1e-3
    while len(out) < n and tries < 5000:
        i, j = rng.integers(0, H - size + 1), rng.integers(0, W - size + 1); tries += 1
        if m[i:i + size, j:j + size].mean() >= min_valid and all(abs(i - a) + abs(j - b) > size // 2 for a, b in out):
            out.append((i, j))
    return out

TEST_CROPS = {'paviaU': 3, 'paviaC': 2, 'urban': 2, 'dcmall': 2, 'botswana': 3, 'muufl64': 2, 'houston13': 2,
              'houston18': 1, 'trento': 2, 'cave_peppers': 2, 'cave_pompoms': 2, 'salinas': 3, 'cuprite': 3,
              **{f'kaist{i:02d}': 1 for i in range(1, 11)}}

def build_test_sets(path='./testsets', size=64):
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(2026)
    index = []
    for name, n in TEST_CROPS.items():
        d = load(name); x = d['x']
        for ci, (i, j) in enumerate(valid_crop_origins(x, size, n, rng)):
            clean = x[i:i + size, j:j + size]
            for cname, kinds in TEST_CASES.items():
                r2 = np.random.default_rng(zlib.crc32(f"{name}|{ci}|{cname}".encode()))
                y, _ = degrade(clean, r2, kinds)
                fn = f'{name}_c{ci}_{cname.split()[0]}.npz'
                np.savez_compressed(f'{path}/{fn}', clean=clean, noisy=y, wl=d['wl'])
                index.append(dict(file=fn, cube=name, crop=ci, case=cname, origin=[int(i), int(j)], sensor=META[name]['sensor']))
    json.dump(index, open(f'{path}/index.json', 'w'), indent=0)
    return index

if __name__ == '__main__':
    idx = build_test_sets(); print(len(idx), 'test items')
    for f in FOLDS: print(f, len(train_pool(f)), 'train cubes')
