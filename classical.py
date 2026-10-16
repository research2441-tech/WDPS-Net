"""Model-based baselines on the fixed test items: BM4D (noise-whitened, blind), FastHyDe and L1HyMixDe (HyDe library
implementations), and LRMR (patch-wise low-rank + sparse matrix recovery via GoDec). Usage: python classical.py BM4D"""
import sys, json, os, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
from wdps_metrics import all_metrics
TS = './testsets'; INDEX = json.load(open(f'{TS}/index.json'))

def band_sigma(y):
    d = (y[0:-1:2, 0:-1:2] - y[1::2, 0:-1:2][:y[0:-1:2, 0:-1:2].shape[0]] - y[0:-1:2, 1::2][:, :y[0:-1:2, 0:-1:2].shape[1]] + y[1::2, 1::2]) / 2
    return np.maximum(np.median(np.abs(d.reshape(-1, y.shape[2])), 0) / 0.6745, 1e-3)

def run_bm4d(y):
    import bm4d
    s = band_sigma(y); z = y / s[None, None, :]
    out = bm4d.bm4d(z.astype(np.float32), 1.0, profile='np')
    return np.asarray(out) * s[None, None, :]

def run_fasthyde(y):
    ok = y.reshape(-1, y.shape[2]).std(0) > 1e-6          # constant (no-data) bands are passed through unchanged
    if not ok.all():
        out = y.copy(); out[:, :, ok] = run_fasthyde(y[:, :, ok]); return out
    from hyde.spatial_domain.fast_hyde import FastHyDe
    k = int(min(10, y.shape[2] - 1))
    s = band_sigma(y)            # band-wise noise whitening, then the i.i.d. FastHyDe path (the non-i.i.d. path of the
    z = (y / s[None, None, :]).astype(np.float64)   # library returns unstable outputs on these crops)
    return FastHyDe()(torch.from_numpy(z), noise_type='additive', iid=True, k_subspace=k).numpy() * s[None, None, :]

def run_l1hymixde(y):
    from hyde.spatial_domain.l1hymixde import L1HyMixDe
    k = int(min(10, y.shape[2] - 1))
    return L1HyMixDe()(torch.from_numpy(y.astype(np.float64)), k_subspace=k, p=0.05, max_iter=10).numpy()

def godec(X, r, card, iters=10):
    L = X.copy(); S = np.zeros_like(X)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X - S, full_matrices=False); L = (U[:, :r] * s[:r]) @ Vt[:r]
        R = X - L; thr = np.partition(np.abs(R).ravel(), -card)[-card] if card > 0 else np.inf
        S = np.where(np.abs(R) >= thr, R, 0)
    return L

def run_lrmr(y, p=20, step=8, r=4, frac=0.08):
    H, W, B = y.shape; out = np.zeros_like(y); cnt = np.zeros((H, W, 1), np.float32)
    ii = sorted(set(list(range(0, H - p + 1, step)) + [H - p])); jj = sorted(set(list(range(0, W - p + 1, step)) + [W - p]))
    for i in ii:
        for j in jj:
            X = y[i:i + p, j:j + p].reshape(-1, B)
            out[i:i + p, j:j + p] += godec(X, r, int(frac * X.size)).reshape(p, p, B); cnt[i:i + p, j:j + p] += 1
    return out / cnt

METHODS = dict(Noisy=lambda y: y, BM4D=run_bm4d, FastHyDe=run_fasthyde, L1HyMixDe=run_l1hymixde, LRMR=run_lrmr)

if __name__ == '__main__':
    name = sys.argv[1]; f = METHODS[name]; os.makedirs('./results', exist_ok=True)
    out_path = f'./results/{name}_classical.json'; part = out_path.replace('.json', '.partial.json')
    done = {r['file']: r for r in json.load(open(part))['items']} if os.path.exists(part) else {}
    torch.set_num_threads(1); res = []
    subset = [it for it in INDEX if it['crop'] == 0 and not (it['cube'].startswith('kaist') and int(it['cube'][5:]) > 3)]
    for it in subset:
        if it['file'] in done: res.append(done[it['file']]); continue
        d = np.load(f"{TS}/{it['file']}"); t = time.time()
        try:
            xh = f(d['noisy'].astype(np.float32))
        except Exception as e:
            print('FAIL', it['file'], e, flush=True); xh = d['noisy']
        r = dict(it, **all_metrics(d['clean'], np.nan_to_num(xh)), sec=time.time() - t, B=int(d['clean'].shape[2])); res.append(r)
        print(name, it['file'], {k: round(r[k], 3) for k in ['psnr', 'ssim', 'sam']}, round(r['sec'], 1), flush=True)
        if len(res) % 5 == 0: json.dump(dict(name=name, items=res), open(part, 'w'))
    json.dump(dict(name=name, items=res), open(out_path, 'w'))
