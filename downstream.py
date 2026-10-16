"""Real-noise and downstream evaluation.
(a) Indian Pines, 220 bands incl. water-absorption bands (real AVIRIS noise): SVM classification OA/AA/kappa.
(b) MUUFL Gulfport, 72 bands incl. noisy edge bands (real CASI noise): ACE detection of emplaced cloth targets.
(c) Cuprite with synthetic mixture noise (C5): ACE detection with the library target signature.
Usage: python downstream.py <method> <ckpt-glob or 'classical'>"""
import sys, os, glob, json, time, numpy as np, torch, scipy.io as sio
import wdps_data as D, evaluate as E, classical as CL
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, cohen_kappa_score

def predict_tiled(m, y, wl, tile=64, ov=8):
    H, W, B = y.shape; out = np.zeros_like(y); cnt = np.zeros((H, W, 1), np.float32); st = tile - ov
    ii = sorted(set(list(range(0, max(1, H - tile + 1), st)) + [max(0, H - tile)]))
    jj = sorted(set(list(range(0, max(1, W - tile + 1), st)) + [max(0, W - tile)]))
    for i in ii:
        for j in jj:
            xh, _ = E.predict(m, y[i:i + tile, j:j + tile], wl); out[i:i + tile, j:j + tile] += xh; cnt[i:i + tile, j:j + tile] += 1
    return out / cnt

def classify(cube, gt, reps=5, frac=0.1):
    X = cube.reshape(-1, cube.shape[2]); g = gt.ravel(); lab = np.where(g > 0)[0]; res = []
    for r in range(reps):
        rng = np.random.default_rng(r); tr = []
        for c in np.unique(g[lab]):
            idx = lab[g[lab] == c]; k = max(3, int(round(frac * len(idx)))); tr += list(rng.choice(idx, k, replace=False))
        tr = np.array(tr); te = np.setdiff1d(lab, tr)
        sc = StandardScaler().fit(X[tr]); clf = SVC(C=100, gamma='scale').fit(sc.transform(X[tr]), g[tr])
        p = clf.predict(sc.transform(X[te])); y = g[te]
        oa = (p == y).mean(); aa = np.mean([(p[y == c] == c).mean() for c in np.unique(y)]); kap = cohen_kappa_score(y, p)
        res.append((oa, aa, kap))
    return np.array(res)

def ace(cube, t, valid=None):
    X = cube.reshape(-1, cube.shape[2]).astype(np.float64)
    V = X if valid is None else X[valid.ravel() > 0]
    mu = V.mean(0); C = np.cov((V - mu).T) + 1e-6 * np.eye(X.shape[1]); Ci = np.linalg.inv(C)
    Xc = X - mu; tc = t - mu
    num = (Xc @ Ci @ tc) ** 2; den = (tc @ Ci @ tc) * np.einsum('ij,jk,ik->i', Xc, Ci, Xc)
    return (num / np.maximum(den, 1e-12)).reshape(cube.shape[:2])

def muufl_detection(cube, d):
    rows, cols, types, sizes = d['t_row'].astype(int) - 1, d['t_col'].astype(int) - 1, d['t_type'], d['t_size']
    valid = d['valid'].astype(bool); H, W = valid.shape
    far = np.ones((H, W), bool)
    for r, c in zip(rows, cols): far[max(0, r - 4):r + 5, max(0, c - 4):c + 5] = False
    aucs = {}
    for ty in ['brown', 'dark green', 'pea green', 'faux vineyard green']:
        sig_idx = [i for i in range(len(types)) if types[i] == ty and sizes[i] >= 3]
        pos_idx = [i for i in range(len(types)) if types[i] == ty and sizes[i] <= 1]
        if not sig_idx or not pos_idx: continue
        t = np.mean([cube[rows[i], cols[i]] for i in sig_idx], 0)
        s = ace(cube, t, valid)
        pos = np.array([s[rows[i], cols[i]] for i in pos_idx]); neg = s[far & valid]
        aucs[ty] = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    return aucs

def cuprite_noisy():
    m = sio.loadmat('./data/TargetDetection/cuprite_ECEM.mat'); X = m['X'].astype(np.float32)
    a, b = np.percentile(X, 0.1), np.percentile(X, 99.9)
    t = np.clip((m['d'].ravel() - a) / (b - a), 0, 1)
    d = D.load('cuprite'); x = d['x']
    y, _ = D.degrade(x, np.random.default_rng(77), D.TEST_CASES['C5 Mixture'])
    return x, y, d['wl'], t, d['gt'].astype(bool)

def run(method, spec):
    out = dict(method=method)
    ip = D.load('indianpines220'); mf = D.load('muufl72'); cx, cy, cwl, ct, cgt = cuprite_noisy()
    if spec == 'classical':
        f = CL.METHODS[method]; dens = lambda m, y, wl: f(y.astype(np.float32))
        models = [None]
    else:
        models = sorted(sum([glob.glob(g) for g in spec.split(',')], [])); dens = lambda m, y, wl: predict_tiled(m, y, wl)
    rows = []
    for mp in models:
        m = None
        if mp is not None: m = E.load_model(mp)[0]
        torch.set_num_threads(int(os.environ.get('WDPS_THREADS', 1)))
        t0 = time.time(); xi = np.clip(dens(m, ip['x'], ip['wl']), 0, 1); ti = time.time() - t0
        cls = classify(xi, ip['gt'])
        rec = dict(model=mp, ip_oa=cls[:, 0].tolist(), ip_aa=cls[:, 1].tolist(), ip_kappa=cls[:, 2].tolist(), ip_sec=ti)
        nb = ip['noisy']; rec['ip_noisyband_rough'] = float(np.abs(np.diff(xi[:, :, nb], axis=0)).mean())
        if spec == 'classical' or 'VNIR' in (mp or ''):
            xm = np.clip(dens(m, mf['x'], mf['wl']), 0, 1); rec['muufl_auc'] = muufl_detection(xm, mf)
        if spec == 'classical' or 'ROSIS' in (mp or ''):
            xc = np.clip(dens(m, cy, cwl), 0, 1); s = ace(xc, ct)
            rec['cuprite_auc'] = float(roc_auc_score(cgt.ravel(), s.ravel()))
        rows.append(rec); print(method, json.dumps({k: (np.mean(v) if isinstance(v, list) else v) for k, v in rec.items() if k != 'model'}, default=str), flush=True)
    out['runs'] = rows
    json.dump(out, open(f'./results/downstream_{method}.json', 'w'))

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
