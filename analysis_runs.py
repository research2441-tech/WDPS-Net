"""Inference-only analyses on trained checkpoints.
  python analysis_runs.py robust <model>      noise-intensity sweep (C5 mixture, sigma scale 0.5..2) on ROSIS+HYDICE folds
  python analysis_runs.py sens                inference-time sensitivity of WDPS-Net (delta_0, subspace ranks, top-k)
  python analysis_runs.py routing             mean routing weights per expert and degradation case (WDPS, seed 0)
  python analysis_runs.py ensemble            seed-ensemble vs. head uncertainty calibration (WDPS, all folds)"""
import sys, os, glob, json, time, zlib, numpy as np, torch
from scipy.stats import norm, spearmanr
import wdps_data as D, evaluate as E, wdps_models as WM
from wdps_metrics import all_metrics
R = './results'
torch.set_num_threads(int(os.environ.get('WDPS_THREADS', 1)))

def fold_items(fold, cases=None):
    return [it for it in E.INDEX if it['cube'] in D.FOLDS[fold] and (cases is None or it['case'][:2] in cases)]

def robust(model):
    out = []
    for fold in ['ROSIS', 'HYDICE']:
        crops = sorted({(it['cube'], it['crop']) for it in fold_items(fold)})
        if model in ('FastHyDe', 'LRMR', 'Noisy'):
            import classical as CL; mods = [(None, CL.METHODS[model])]
        else:
            mods = [(p, E.load_model(p)[0]) for p in sorted(glob.glob(f'./ckpt/{model}_{fold}_s?.pt'))]
        for cube, ci in crops:
            ref = [it for it in fold_items(fold) if it['cube'] == cube and it['crop'] == ci][0]
            clean = np.load(f"{E.TS}/{ref['file']}")['clean']; wl = np.load(f"{E.TS}/{ref['file']}")['wl']
            for sc in [0.5, 1.0, 1.5, 2.0]:
                rng = np.random.default_rng(zlib.crc32(f'rob|{cube}|{ci}|{sc}'.encode()))
                y, _ = D.degrade(clean, rng, D.TEST_CASES['C5 Mixture'], dict(sig_scale=sc))
                ps = []
                for p, m in mods:
                    xh = m(y) if p is None else E.predict(m, y, wl)[0]
                    ps.append(all_metrics(clean, xh))
                out.append(dict(fold=fold, cube=cube, crop=ci, scale=sc, psnr=float(np.mean([q['psnr'] for q in ps])),
                                sam=float(np.mean([q['sam'] for q in ps])), noisy=all_metrics(clean, y)['psnr']))
        print(model, fold, flush=True)
    json.dump(out, open(f'{R}/robust_{model}.json', 'w'))

def sens():
    settings = [('delta0', 5.0), ('delta0', 10.0), ('delta0', 20.0), ('ranks', (2, 6)), ('ranks', (4, 10)), ('ranks', (8, 16)),
                ('topk', 1), ('topk', 2), ('topk', 3)]
    out = []
    for fold in ['ROSIS', 'HYDICE']:
        items = fold_items(fold, ['C1', 'C5', 'C6'])
        for p in sorted(glob.glob(f'./ckpt/WDPS_{fold}_s?.pt')):
            m = E.load_model(p)[0]
            for name, v in settings:
                WM.DLAM_REF, WM.RANKS = 10.0, (4, 10); m.moe1.topk = m.moe2.topk = 2
                if name == 'delta0': WM.DLAM_REF = v
                if name == 'ranks': WM.RANKS = v
                if name == 'topk': m.moe1.topk = m.moe2.topk = v
                for it in items:
                    d = np.load(f"{E.TS}/{it['file']}"); xh, _ = E.predict(m, d['noisy'], d['wl'])
                    out.append(dict(fold=fold, model=os.path.basename(p), setting=name, value=str(v), case=it['case'][:2],
                                    file=it['file'], **all_metrics(d['clean'], xh)))
            print(p, flush=True)
    WM.DLAM_REF, WM.RANKS = 10.0, (4, 10)
    json.dump(out, open(f'{R}/sens_WDPS.json', 'w'))

def routing():
    out = []
    for fold in D.FOLDS:
        p = f"./ckpt/WDPS_{fold.replace('/', '-')}_s0.pt"
        if not os.path.exists(p): continue
        m = E.load_model(p)[0]
        for it in fold_items(fold):
            d = np.load(f"{E.TS}/{it['file']}")
            with torch.no_grad():
                o = m(torch.from_numpy(d['noisy']).permute(2, 0, 1)[None].float(), torch.from_numpy(d['wl'])[None].float())
            g = o['gates'][0].numpy()            # (B, K) first routed layer
            st = o['stats'][0].numpy()           # (B, 5) clean-free descriptors
            out.append(dict(fold=fold, case=it['case'][:2], cube=it['cube'], gates=g.mean(0).tolist(),
                            gates_band=g.tolist(), stats=st.tolist(), wl=d['wl'].tolist()))
        print(fold, flush=True)
    json.dump(out, open(f'{R}/routing_WDPS.json', 'w'))

def ensemble():
    out = []
    for fold in D.FOLDS:
        f = fold.replace('/', '-')
        mods = [E.load_model(p)[0] for p in sorted(glob.glob(f'./ckpt/WDPS_{f}_s?.pt'))]
        if len(mods) < 3: continue
        for it in fold_items(fold):
            d = np.load(f"{E.TS}/{it['file']}"); x = d['clean']
            preds = [E.predict(m, d['noisy'], d['wl']) for m in mods]
            mu = np.mean([np.clip(a, 0, 1) for a, _ in preds], 0); err = mu - x
            v_head = np.mean([np.exp(lv) for _, lv in preds], 0); v_ens = np.var([np.clip(a, 0, 1) for a, _ in preds], 0)
            rec = dict(fold=fold, case=it['case'][:2], file=it['file'], psnr_ens=all_metrics(x, mu)['psnr'])
            for nm, v in [('head', v_head), ('ens', v_ens + 1e-8), ('both', v_head + v_ens)]:
                zz = np.abs(err) / np.sqrt(v)
                cov = {str(q): float((zz <= norm.ppf(0.5 + q / 2)).mean()) for q in [0.5, 0.8, 0.9, 0.95]}
                bs = np.sqrt(v.reshape(-1, v.shape[2]).mean(0)); br = np.sqrt((err ** 2).reshape(-1, err.shape[2]).mean(0))
                rec[nm] = dict(cov=cov, ce=float(np.mean([abs(float(q) - c) for q, c in cov.items()])),
                               rho=float(spearmanr(bs, br).correlation), nll=float((0.5 * err ** 2 / v + 0.5 * np.log(2 * np.pi * v)).mean()))
            out.append(rec)
        print(fold, flush=True)
    json.dump(out, open(f'{R}/ensemble_WDPS.json', 'w'))

if __name__ == '__main__':
    {'robust': lambda: robust(sys.argv[2]), 'sens': sens, 'routing': routing, 'ensemble': ensemble}[sys.argv[1]]()
