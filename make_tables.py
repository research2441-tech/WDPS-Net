"""Aggregate result JSONs into tables and summary numbers."""
from res_utils import *
S = {}
T = {}
# ---------------- main table: PSNR per held-out fold, cross-scene and overall
rows = [['Method', 'Params'] + [FLAB[f] for f in FOLDS] + ['AVIRIS cross-scene', 'Held-out mean']]
eff = json.load(open(f'{R}/efficiency.json')) if os.path.exists(f'{R}/efficiency.json') else None
best = {}
fs_all = {m: fold_seed_means(m) for m in DEEP}
for m in DEEP:
    v = fs_all[m]
    S.setdefault('fold_psnr', {})[m] = {k: list(map(float, x)) for k, x in v.items()}
noisy_items = load('Noisy_classical')
for col in FOLDS + ['X']:
    best[col] = max(DEEP, key=lambda m: np.mean(fs_all[m][col]) if fs_all[m][col] else -1)
for m in DEEP:
    v = fs_all[m]
    allheld = [np.mean([v[f][s] for f in FOLDS]) for s in range(min(len(v[f]) for f in FOLDS))] if all(v[f] for f in FOLDS) else []
    S.setdefault('heldout_mean', {})[m] = list(map(float, allheld))
    par = '-' if eff is None else f"{[r for r in eff['rows'] if r['model'] == m][0]['params'] / 1e3:.1f}k"
    cells = []
    for col in FOLDS + ['X']:
        s = ms(v[col]); cells.append(f'**{s}**' if best[col] == m and s != '-' else s)
    rows.append([NAMES[m], par] + cells + [ms(allheld)])
T['main'] = rows

# ---------------- per-case table (held-out items, seed-averaged) with SSIM/SAM/ERGAS overall
pi = {m: per_item(m) for m in DEEP}
rows = [['Method'] + CASES + ['MSSIM', 'SAM', 'ERGAS']]
for m in DEEP:
    it = pi[m]
    if not it: continue
    cells = [f"{np.mean([r['psnr'] for r in it.values() if r['case'] == c]):.2f}" for c in CASES]
    rows.append([NAMES[m]] + cells + [f"{np.mean([r['ssim'] for r in it.values()]):.4f}", f"{np.mean([r['sam'] for r in it.values()]):.2f}",
                                     f"{np.mean([r['ergas'] for r in it.values()]):.2f}"])
    S.setdefault('per_case', {})[m] = {c: float(np.mean([r['psnr'] for r in it.values() if r['case'] == c])) for c in CASES}
    S.setdefault('overall', {})[m] = {k: float(np.mean([r[k] for r in it.values()])) for k in MET}
T['cases'] = rows

# ---------------- statistics: WDPS vs each deep competitor over held-out items (seed-averaged PSNR)
st_rows = [['Comparison', 'Mean $\\Delta$PSNR (dB)', 'Wins/ties/losses', 'Wilcoxon $p$ (Holm)', "Cliff's $\\delta$", 'Mean $\\Delta$SAM (°)']]
comp = [m for m in DEEP if m != 'WDPS']
ps, recs = [], []
for m in comp:
    keys = sorted(set(pi['WDPS']) & set(pi[m]))
    if len(keys) < 10: continue
    a = np.array([pi['WDPS'][k]['psnr'] for k in keys]); b = np.array([pi[m][k]['psnr'] for k in keys])
    sa = np.array([pi['WDPS'][k]['sam'] for k in keys]); sb = np.array([pi[m][k]['sam'] for k in keys])
    d = a - b; p = wilcoxon(a, b).pvalue
    recs.append((m, d, p, cliff(a, b), sa - sb, len(keys))); ps.append(p)
adj = holm(np.array(ps)) if ps else []
for (m, d, p, cd, ds, n), pa in zip(recs, adj):
    w, t, l = int((d > 0.05).sum()), int((abs(d) <= 0.05).sum()), int((d < -0.05).sum())
    st_rows.append([f"WDPS-Net vs. {NAMES[m]}", f'{d.mean():+.2f}', f'{w}/{t}/{l}', f'{pa:.1e}', f'{cd:+.2f}', f'{ds.mean():+.2f}'])
    S.setdefault('stats', {})[m] = dict(dpsnr=float(d.mean()), wins=w, ties=t, losses=l, p_holm=float(pa), cliff=cd, dsam=float(ds.mean()), n=n)
# Friedman over methods
keys = sorted(set.intersection(*[set(pi[m]) for m in DEEP])) if all(pi.values()) else []
if len(keys) > 10:
    X = np.array([[pi[m][k]['psnr'] for m in DEEP] for k in keys])
    fr = friedmanchisquare(*X.T); rk = np.mean([rankdata(-r) for r in X], 0)
    S['friedman'] = dict(stat=float(fr.statistic), p=float(fr.pvalue), ranks={m: float(r) for m, r in zip(DEEP, rk)}, n=len(keys))
T['stats'] = st_rows

# ---------------- classical subset (96 items): deep seed-averaged vs model-based
cl = {c: load(f'{c}_classical') for c in ['Noisy', 'BM4D', 'FastHyDe', 'L1HyMixDe', 'LRMR']}
sub_files = [r['file'] for r in (cl['Noisy'] or {'items': []})['items']]
if sub_files:
    allx = {m: {**per_item(m), **per_item(m, xscene=True)} for m in DEEP}
    rows = [['Method', 'MPSNR', 'MSSIM', 'SAM', 'ERGAS', 'C1', 'C4', 'C5', 'C6', 'Time (s)']]
    for c in ['Noisy', 'BM4D', 'FastHyDe', 'L1HyMixDe', 'LRMR']:
        if cl[c] is None: continue
        it = {r['file']: r for r in cl[c]['items']}
        fl = [f for f in sub_files if f in it]
        if len(fl) < len(sub_files): continue
        rows.append([NAMES[c]] + [f"{np.mean([it[f][k] for f in fl]):.{4 if k == 'ssim' else 2}f}" for k in MET] +
                    [f"{np.mean([it[f]['psnr'] for f in fl if it[f]['case'].startswith(c5)]):.2f}" for c5 in ['C1', 'C4', 'C5', 'C6']] +
                    [f"{np.mean([it[f]['sec'] for f in fl]):.1f}" if c != 'Noisy' else '-'])
        S.setdefault('classical', {})[c] = {k: float(np.mean([it[f][k] for f in fl])) for k in MET + ['sec']}
    for m in DEEP:
        it = allx[m]; fl = [f for f in sub_files if f in it]
        if len(fl) < len(sub_files): continue
        secs = []
        for f in FOLDS:
            for s in range(3):
                for r in (run_items(m, f, s) or []):
                    if r['file'] in sub_files: secs.append(r['sec'])
        rows.append([NAMES[m]] + [f"{np.mean([it[f][k] for f in fl]):.{4 if k == 'ssim' else 2}f}" for k in MET] +
                    [f"{np.mean([it[f]['psnr'] for f in fl if it[f]['case'] == c5]):.2f}" for c5 in ['C1', 'C4', 'C5', 'C6']] + [f'{np.mean(secs):.1f}'])
        S.setdefault('classical', {})[m] = {k: float(np.mean([it[f][k] for f in fl])) for k in MET}
    T['classical'] = rows

# ---------------- ablation (seed 0, ROSIS + HYDICE folds, held-out items)
ABL = [('', 'Full WDPS-Net'), ('-kappa1', 'fixed gap exponent ($\\kappa_c=1$)'), ('-wave', 'band index instead of wavelength codes'), ('-gap', 'index stepping ($\\kappa_c=0$)'),
       ('-prompt', 'without degradation prompt'), ('-route', 'uniform expert mixture (no routing)'),
       ('-subspace', 'without unmixing-subspace prior'), ('-specSSM', 'spectral convolution instead of spectral scan')]
full = {}
for f in ['ROSIS', 'HYDICE']:
    for r in (run_items('WDPS', f, 0) or []):
        if r['cube'] not in D.XSCENE: full[r['file']] = r
rows = [['Variant', 'ROSIS', 'HYDICE', 'C5 (both)', 'C6 (both)', 'SAM', '$\\Delta$ vs. full', '$p$ (Holm)']]
ab_ps, ab_rec = [], []
for tag, lab in ABL:
    its = {}
    for f in ['ROSIS', 'HYDICE']:
        for r in (run_items('WDPS', f, 0, tag) or []):
            if r['cube'] not in D.XSCENE: its[r['file']] = r
    if not its: continue
    fr = lambda f: np.mean([r['psnr'] for r in its.values() if r['cube'] in D.FOLDS[f]])
    keys = sorted(set(its) & set(full)); d = np.array([its[k]['psnr'] - full[k]['psnr'] for k in keys])
    p = wilcoxon(d).pvalue if tag and np.any(d != 0) else 1.0
    ab_rec.append([lab, f"{fr('ROSIS'):.2f}", f"{fr('HYDICE'):.2f}",
                   f"{np.mean([r['psnr'] for r in its.values() if r['case'].startswith('C5')]):.2f}",
                   f"{np.mean([r['psnr'] for r in its.values() if r['case'].startswith('C6')]):.2f}",
                   f"{np.mean([r['sam'] for r in its.values()]):.2f}", '-' if not tag else f'{d.mean():+.2f}'])
    ab_ps.append(p if tag else None)
    S.setdefault('ablation', {})[tag or 'full'] = dict(rosis=float(fr('ROSIS')), hydice=float(fr('HYDICE')), delta=float(d.mean()) if tag else 0.0, p=float(p))
ps_only = [p for p in ab_ps if p is not None]
adj = list(holm(np.array(ps_only))) if ps_only else []
for rec, p in zip(ab_rec, ab_ps):
    rec.append('-' if p is None else f'{adj.pop(0):.1e}'); rows.append(rec)
T['ablation'] = rows
# SSM base equals the "remove everything" corner of the ablation
# ---------------- SSA
rows = [['Model', 'Adaptation'] + [FLAB[f] for f in FOLDS] + ['Mean']]
for m in ['QRNN3D', 'WDPS']:
    for tag, lab in [('', 'none'), ('+SSA', 'label-free, 150 it.')] + ([('+SSA-um', 'without $\\mathcal{L}_{\\mathrm{um}}$')] if m == 'WDPS' else []):
        vals, cells = [], []
        for f in FOLDS:
            seeds = (0,)
            v = []
            for s in seeds:
                d = load(f'{m}_{FS[f]}_s{s}{tag}')
                if d: v.append(np.mean([r['psnr'] for r in d['items'] if r['cube'] not in D.XSCENE]))
            cells.append(ms(v)); vals.append(np.mean(v) if v else np.nan)
        if all(np.isnan(vals)): continue
        rows.append([NAMES[m], lab] + cells + [f'{np.nanmean(vals):.2f}'])
        S.setdefault('ssa', {})[f'{m}{tag}'] = [float(x) for x in vals]
T['ssa'] = rows

json.dump(S, open(f'{R}/summary.json', 'w'), indent=1)
json.dump(T, open(f'{R}/tables.json', 'w'), indent=1)
for k, v in T.items():
    print('==', k)
    for r in v: print('  ', ' | '.join(map(str, r)))

# ---------------- learned gap exponents
import torch
ks = []
for f in FOLDS:
    for s_ in range(3):
        p = f"./ckpt/WDPS_{FS[f]}_s{s_}.pt"
        if os.path.exists(p):
            st = torch.load(p, weights_only=False)['state']
            for k, v in st.items():
                if k.endswith('kappa'): ks.append((k.split('.')[0], torch.sigmoid(v).numpy()))
if ks:
    allk = np.concatenate([v for _, v in ks])
    S['kappa'] = dict(mean=float(allk.mean()), frac_gt_half=float((allk > 0.5).mean()), q10=float(np.quantile(allk, 0.1)),
                      q90=float(np.quantile(allk, 0.9)), per_layer={L: float(np.mean([v.mean() for l, v in ks if l == L])) for L in ['e1', 'e2', 'm1', 'd2', 'd1']})

# ---------------- downstream
DS = ['Noisy', 'FastHyDe', 'LRMR'] + DEEP
rows = [['Method', 'IP OA (%)', 'IP AA (%)', 'IP $\\kappa$', 'MUUFL AUC', 'Cuprite AUC (C5)']]
for m in DS:
    d = load(f'downstream_{m}')
    if not d: continue
    runs = d['runs']
    oa = [np.mean(r['ip_oa']) for r in runs]; aa = [np.mean(r['ip_aa']) for r in runs]; kp = [np.mean(r['ip_kappa']) for r in runs]
    mu = [np.mean(list(r['muufl_auc'].values())) for r in runs if 'muufl_auc' in r]
    cu = [r['cuprite_auc'] for r in runs if 'cuprite_auc' in r]
    rows.append([NAMES[m], ms(100 * np.array(oa)), ms(100 * np.array(aa)), ms(np.array(kp), 3), ms(np.array(mu), 3), ms(np.array(cu), 3)])
    S.setdefault('downstream', {})[m] = dict(oa=float(np.mean(oa)), aa=float(np.mean(aa)), kappa=float(np.mean(kp)),
                                             muufl=float(np.mean(mu)) if mu else None, cuprite=float(np.mean(cu)) if cu else None)
T['downstream'] = rows

# ---------------- efficiency
if os.path.exists(f'{R}/efficiency.json'):
    E_ = json.load(open(f'{R}/efficiency.json'))['rows']
    rows = [['Method', 'Params (k)', 'GMAC (B=31)', 'GMAC (B=204)', 'Latency B=31 (ms)', 'Latency B=204 (ms)', 'Peak memory B=204 (MB)', 'Train time (min)']]
    for m in DEEP:
        r31 = [r for r in E_ if r['model'] == m and r['B'] == 31][0]; r204 = [r for r in E_ if r['model'] == m and r['B'] == 204][0]
        tt = [load(f'{m}_{FS[f]}_s{s_}') for f in FOLDS for s_ in range(3)]
        tt = [x['train_time'] / 60 for x in tt if x]
        rows.append([NAMES[m], f"{r31['params'] / 1e3:.1f}", f"{r31['gmacs']:.2f}", f"{r204['gmacs']:.2f}", f"{r31['ms']:.0f}", f"{r204['ms']:.0f}",
                     f"{r204['peak_mb']:.0f}", f'{np.mean(tt):.1f}' if tt else '-'])
        S.setdefault('eff', {})[m] = dict(params=r31['params'], g31=r31['gmacs'], g204=r204['gmacs'], ms31=r31['ms'], ms204=r204['ms'], mem=r204['peak_mb'], train_min=float(np.mean(tt)) if tt else None)
    T['eff'] = rows

# ---------------- uncertainty
if os.path.exists(f'{R}/ensemble_WDPS.json'):
    e = json.load(open(f'{R}/ensemble_WDPS.json'))
    rows = [['Uncertainty source', 'Cov. 50%', 'Cov. 80%', 'Cov. 90%', 'Cov. 95%', 'CE', 'Spearman $\\rho$ (bands)', 'NLL']]
    for nm, lab in [('head', 'variance head (single model)'), ('ens', 'three-seed ensemble spread'), ('both', 'head + ensemble')]:
        c = {q: np.mean([r[nm]['cov'][q] for r in e]) for q in ['0.5', '0.8', '0.9', '0.95']}
        rows.append([lab] + [f'{c[q]:.3f}' for q in c] + [f"{np.mean([r[nm]['ce'] for r in e]):.3f}", f"{np.nanmean([r[nm]['rho'] for r in e]):.3f}", f"{np.mean([r[nm]['nll'] for r in e]):.2f}"])
        S.setdefault('unc', {})[nm] = dict(cov={'q' + str(int(round(float(k) * 100))): v for k, v in c.items()}, ce=float(np.mean([r[nm]['ce'] for r in e])), rho=float(np.nanmean([r[nm]['rho'] for r in e])), nll=float(np.mean([r[nm]['nll'] for r in e])))
    T['unc'] = rows

json.dump(S, open(f'{R}/summary.json', 'w'), indent=1)
json.dump(T, open(f'{R}/tables.json', 'w'), indent=1)

# ---------------- derived numbers for the text
H = S.get('heldout_mean', {})
if H.get('WDPS'):
    S['gain'] = {m: float(np.mean(H['WDPS']) - np.mean(H[m])) for m in DEEP if H.get(m)}
    fp = S['fold_psnr']
    S['fold_gain_vs_best_baseline'] = {}
    for f in FOLDS + ['X']:
        bb = max([m for m in DEEP if m != 'WDPS' and fp[m][f]], key=lambda m: np.mean(fp[m][f]))
        S['fold_gain_vs_best_baseline'][f.replace('/', '-')] = dict(baseline=bb, gain=float(np.mean(fp['WDPS'][f]) - np.mean(fp[bb][f])),
                                                                   wdps=float(np.mean(fp['WDPS'][f])), base=float(np.mean(fp[bb][f])))
    S['fold_mean'] = {m: {f.replace('/', '-'): float(np.mean(fp[m][f])) if fp[m][f] else None for f in FOLDS + ['X']} for m in DEEP}
    S['heldout_mean_avg'] = {m: float(np.mean(H[m])) for m in DEEP if H.get(m)}
    S['heldout_mean_sd'] = {m: float(np.std(H[m])) for m in DEEP if H.get(m)}
json.dump(S, open(f'{R}/summary.json', 'w'), indent=1)
