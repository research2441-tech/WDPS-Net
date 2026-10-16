"""Aggregate result JSONs into summary statistics, LaTeX-free table rows for the document builder, and numbers for the text."""
import json, glob, os, re, numpy as np, itertools
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import wdps_data as D
R = './results'; OUT = './paper'
FOLDS = list(D.FOLDS); FS = {f: f.replace('/', '-') for f in FOLDS}
FLAB = {'ROSIS': 'ROSIS', 'HYDICE': 'HYDICE', 'Hyperion': 'Hyperion', 'VNIR-CASI/AISA': 'CASI/AISA', 'Natural': 'CAVE/KAIST'}
DEEP = ['SqueezeHSI', 'PromptIR3D', 'SSTLite', 'QRNN3D', 'SSMBase', 'WDPS']
NAMES = {'SqueezeHSI': 'SqueezeNet-HSI [@squeezenet_hsi]', 'PromptIR3D': 'PromptIR-3D [@potlapalli2023promptir]', 'SSTLite': 'SST-style [@sst]',
         'QRNN3D': 'QRNN3D [@qrnn3d]', 'SSMBase': 'SSM base [@ssumamba]', 'WDPS': '**WDPS-Net**', 'Noisy': 'Noisy input',
         'BM4D': 'BM4D [@bm4d]', 'FastHyDe': 'FastHyDe [@fasthyde]', 'L1HyMixDe': 'L1HyMixDe [@l1hymixde]', 'LRMR': 'LRMR [@lrmr]'}
CASES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
MET = ['psnr', 'ssim', 'sam', 'ergas']

def load(name):
    p = f'{R}/{name}.json'
    return json.load(open(p)) if os.path.exists(p) and open(p).read(1) == '{' else None

def run_items(model, fold, seed, tag=''):
    d = load(f'{model}{tag}_{FS[fold]}_s{seed}')
    return None if d is None else d['items']

def per_item(model, tag='', seeds=(0, 1, 2), folds=FOLDS, xscene=False):
    """file -> {metric: seed-mean} for held-out items (xscene=False) or cross-scene items averaged over folds and seeds."""
    acc = {}
    for f in folds:
        for s in seeds:
            it = run_items(model, f, s, tag)
            if it is None: continue
            for r in it:
                isx = r['cube'] in D.XSCENE
                if isx != xscene: continue
                key = r['file']
                acc.setdefault(key, {'case': r['case'][:2], 'cube': r['cube'], 'fold': f, **{m: [] for m in MET}})
                for m in MET: acc[key][m].append(r[m])
    return {k: {**{m: float(np.mean(v[m])) for m in MET}, 'case': v['case'], 'cube': v['cube'], 'fold': v['fold'], 'n': len(v['psnr'])}
            for k, v in acc.items()}

def fold_seed_means(model, metric='psnr', tag='', seeds=(0, 1, 2)):
    """{fold: [seed means]} plus 'X' for cross-scene (mean over all fold models of a seed)."""
    out = {}
    for f in FOLDS:
        vals = []
        for s in seeds:
            it = run_items(model, f, s, tag)
            if it: vals.append(np.mean([r[metric] for r in it if r['cube'] not in D.XSCENE]))
        out[f] = vals
    xs = []
    for s in seeds:
        v = [r[metric] for f in FOLDS for r in (run_items(model, f, s, tag) or []) if r['cube'] in D.XSCENE]
        if v: xs.append(np.mean(v))
    out['X'] = xs
    return out

def ms(v, d=2):
    return '-' if not len(v) else (f'{np.mean(v):.{d}f}' if len(v) == 1 else f'{np.mean(v):.{d}f}$\\pm${np.std(v):.{d}f}')

def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.sign(a[:, None] - b[None, :])))

def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(idx):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

