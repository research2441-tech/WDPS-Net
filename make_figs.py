"""Result figures (grayscale). Usage: python make_figs.py fig3 fig5 ..."""
import sys, os, json, glob, numpy as np, torch
from fig_style import *
import wdps_data as D, evaluate as E
from res_utils import fold_seed_means, per_item, FOLDS, FLAB, DEEP, load, run_items
R = './results'
SHORT = {'SqueezeHSI': 'SqueezeNet-HSI', 'PromptIR3D': 'PromptIR-3D', 'SSTLite': 'SST-style', 'QRNN3D': 'QRNN3D', 'SSMBase': 'SSM base',
         'WDPS': 'WDPS-Net', 'FastHyDe': 'FastHyDe', 'LRMR': 'LRMR', 'Noisy': 'Noisy', 'BM4D': 'BM4D', 'L1HyMixDe': 'L1HyMixDe'}
STY = {'SqueezeHSI': ('0.65', 'v'), 'PromptIR3D': ('0.55', 'D'), 'SSTLite': ('0.45', '^'), 'QRNN3D': ('0.30', 's'),
       'SSMBase': ('0.15', 'o'), 'WDPS': ('0.0', '*'), 'FastHyDe': ('0.5', 'P'), 'LRMR': ('0.7', 'X')}

def fig3():
    cols = FOLDS + ['X']; labs = [FLAB[f] for f in FOLDS] + ['AVIRIS\ncross-scene']
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    n = len(DEEP); w = 0.11
    for k, m in enumerate(DEEP):
        v = fold_seed_means(m)
        mu = [np.mean(v[c]) if v[c] else np.nan for c in cols]; sd = [np.std(v[c]) if v[c] else 0 for c in cols]
        x = np.arange(len(cols)) + (k - (n - 1) / 2) * w
        col, mk = STY[m]
        ax.errorbar(x, mu, yerr=sd, fmt=mk, color=col, mfc=col if m != 'WDPS' else '0.0', mec='0.0', mew=0.4,
                    ms=6.5 if m == 'WDPS' else 4.2, elinewidth=0.7, capsize=1.6, label=SHORT[m], zorder=3 if m == 'WDPS' else 2)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs)
    for i in range(len(cols) - 1): ax.axvline(i + 0.5, color='0.85', lw=0.5, zorder=0)
    ax.axvline(len(cols) - 1.5, color='0.4', lw=0.7, ls=':', zorder=0)
    ax.set_ylabel('MPSNR (dB), mean $\\pm$ s.d. over 3 seeds'); ax.grid(axis='x', visible=False)
    ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.2), handletextpad=0.3, columnspacing=1.1)
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.9, y1)
    ax.text(len(cols) - 1, y0 - 0.75, 'seen sensor,\nunseen scenes', ha='center', va='bottom', fontsize=6, color='0.3')
    ax.text(2.0, y0 - 0.75, 'held-out sensor families (leave-one-sensor-out)', ha='center', va='bottom', fontsize=6, color='0.3')
    plt.savefig(f'{FIG}/fig3_folds.png')

def predict_item(model, fold, file, seed=0):
    p = f"./ckpt/{model}_{fold.replace('/', '-')}_s{seed}.pt"
    m = E.load_model(p)[0]; d = np.load(f'{E.TS}/{file}')
    return np.clip(E.predict(m, d['noisy'], d['wl'])[0], 0, 1)

def fig4(file='urban_c0_C5.npz', fold='HYDICE', band=None):
    torch.set_num_threads(2)
    d = np.load(f'{E.TS}/{file}'); x, y, wl = d['clean'], d['noisy'], d['wl']
    import classical as CL
    meth = ['FastHyDe', 'QRNN3D', 'SSTLite', 'SSMBase', 'WDPS']
    est = {}
    for m in meth:
        est[m] = np.clip(CL.METHODS[m](y), 0, 1) if m == 'FastHyDe' else predict_item(m, fold, file)
    b = band if band is not None else int(np.argmax(np.abs(y - x).mean((0, 1)) > 0) * 0 + x.shape[2] // 3)
    from wdps_metrics import mpsnr, sam
    cols = [('Clean', x), ('Noisy', np.clip(y, 0, 1))] + [(SHORT[m], est[m]) for m in meth]
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(2, len(cols) + 1, width_ratios=[1] * len(cols) + [1.9], wspace=0.06, hspace=0.30, left=0.01, right=0.93, top=0.86, bottom=0.13)
    vmax = np.percentile(x[:, :, b], 99.5)
    for j, (nm, im) in enumerate(cols):
        a = fig.add_subplot(gs[0, j]); a.imshow(im[:, :, b], cmap='gray', vmin=0, vmax=vmax); a.set_xticks([]); a.set_yticks([])
        t = nm if j < 2 else f'{nm}\n{mpsnr(x, im):.2f} dB'
        if j == 1: t = f'Noisy\n{mpsnr(x, np.clip(y, 0, 1)):.2f} dB'
        a.set_title(t, fontsize=6.3, pad=2)
        a2 = fig.add_subplot(gs[1, j]); a2.set_xticks([]); a2.set_yticks([])
        if j == 0:
            a2.axis('off'); a2.text(0.5, 0.5, f'band {b + 1}\n$\\lambda$ = {wl[b]:.0f} nm\n\n|error| maps,\nscale 0-0.15', ha='center', va='center', fontsize=6.0, transform=a2.transAxes)
        else:
            a2.imshow(np.abs(im[:, :, b] - x[:, :, b]), cmap='gray_r', vmin=0, vmax=0.15)
            a2.set_xlabel(f'SAM {sam(x, im):.1f}$^\\circ$', fontsize=6.0, labelpad=1.5)
    a = fig.add_subplot(gs[:, -1])
    i0, j0 = 20, 40
    a.plot(wl, x[i0, j0], color='0.0', lw=1.4, label='clean')
    a.plot(wl, np.clip(y[i0, j0], 0, 1), color='0.82', lw=0.5, label='noisy')
    a.plot(wl, est['QRNN3D'][i0, j0], color='0.45', lw=0.9, ls='--', label='QRNN3D')
    a.plot(wl, est['SSMBase'][i0, j0], color='0.3', lw=0.9, ls=':', label='SSM base')
    a.plot(wl, est['WDPS'][i0, j0], color='0.0', lw=0.9, ls='-.', label='WDPS-Net')
    a.set_xlabel('Wavelength (nm)'); a.set_ylabel('Reflectance (scaled)'); a.yaxis.set_label_position('right'); a.yaxis.tick_right()
    lo_, hi_ = x[i0, j0].min(), x[i0, j0].max(); pad = 0.25 * (hi_ - lo_) + 0.03
    a.set_ylim(lo_ - pad, hi_ + 1.6 * pad)
    a.set_title(f'Spectrum at pixel ({i0},{j0})', fontsize=6.3, pad=2)
    a.legend(fontsize=5.0, loc='upper left', ncol=2, handlelength=1.6, columnspacing=0.8, borderaxespad=0.2, frameon=True, framealpha=1.0, edgecolor='0.8', facecolor='white')
    plt.savefig(f'{FIG}/fig4_visual.png')


def fig5():
    """(a) noise-intensity sweep, (b)-(d) inference-time sensitivity of WDPS-Net."""
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw=dict(width_ratios=[1.4, 1, 1, 1], wspace=0.5))
    a = axs[0]
    for m in ['LRMR', 'FastHyDe', 'SqueezeHSI', 'PromptIR3D', 'SSTLite', 'QRNN3D', 'SSMBase', 'WDPS']:
        if not os.path.exists(f'{R}/robust_{m}.json'): continue
        d = json.load(open(f'{R}/robust_{m}.json'))
        sc = sorted({r['scale'] for r in d}); mu = [np.mean([r['psnr'] for r in d if r['scale'] == s]) for s in sc]
        col, mk = STY.get(m, ('0.5', 'o'))
        a.plot(sc, mu, marker=mk, color=col, ms=4.5 if m != 'WDPS' else 6.5, lw=1.0 if m != 'WDPS' else 1.4, label=SHORT[m],
               mfc=col if m != 'WDPS' else '0.0', ls='--' if m in ('LRMR', 'FastHyDe') else '-')
    nz = json.load(open(f'{R}/robust_WDPS.json')) if os.path.exists(f'{R}/robust_WDPS.json') else []
    if nz:
        sc = sorted({r['scale'] for r in nz}); a.plot(sc, [np.mean([r['noisy'] for r in nz if r['scale'] == s]) for s in sc], color='0.8', lw=0.8, label='Noisy')
    a.set_xlabel('Gaussian level scale (C5)'); a.set_ylabel('MPSNR (dB)'); a.set_title('(a) noise intensity', fontsize=7)
    h, l = a.get_legend_handles_labels(); fig.legend(h, l, fontsize=5.4, ncol=9, loc='lower center', bbox_to_anchor=(0.5, -0.01), handlelength=2.0, columnspacing=1.0, markerscale=0.7, frameon=False); fig.subplots_adjust(bottom=0.33)
    s = json.load(open(f'{R}/sens_WDPS.json')) if os.path.exists(f'{R}/sens_WDPS.json') else []
    for ax, (key, lab, conv) in zip(axs[1:], [('delta0', '$\\delta_0$ (nm)', float), ('ranks', 'subspace ranks $r$', str), ('topk', 'experts kept per band', int)]):
        rows = [r for r in s if r['setting'] == key]
        if not rows: continue
        vals = sorted({r['value'] for r in rows}, key=lambda v: (conv(v) if conv is not str else len(v), v))
        for case, mk, col in [('C1', 'o', '0.0'), ('C5', 's', '0.35'), ('C6', '^', '0.6')]:
            mu = [np.mean([r['psnr'] for r in rows if r['value'] == v and r['case'] == case]) for v in vals]
            ax.plot(range(len(vals)), mu, marker=mk, color=col, ms=4, lw=1.0, label=case)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([v.replace('(', '').replace(')', '').replace(' ', '') for v in vals], fontsize=6.3)
        ax.set_xlabel(lab); ax.set_title({'delta0': '(b) reference spacing', 'ranks': '(c) subspace ranks', 'topk': '(d) routing sparsity'}[key], fontsize=7)
        ax.margins(x=0.18); ax.yaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    axs[1].set_ylabel('MPSNR (dB)'); axs[3].legend(fontsize=5.6, loc='best', handlelength=1.2)
    plt.savefig(f'{FIG}/fig5_robust_sens.png')

def fig6():
    """Uncertainty: (a) reliability of central intervals, (b) predicted vs. true band RMSE."""
    e = json.load(open(f'{R}/ensemble_WDPS.json'))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), gridspec_kw=dict(wspace=0.62))
    q = [0.5, 0.8, 0.9, 0.95]; a = axs[0]
    a.plot([0.4, 1], [0.4, 1], color='0.7', lw=0.8, ls=':')
    for nm, lab, mk, col in [('head', 'variance head', 'o', '0.0'), ('ens', 'seed ensemble', 's', '0.55'), ('both', 'head + ensemble', '^', '0.3')]:
        cov = [np.mean([r[nm]['cov'][str(x)] for r in e]) for x in q]
        a.plot(q, cov, marker=mk, color=col, ms=3.8, lw=1.0, label=lab)
    a.set_xlabel('nominal coverage'); a.set_ylabel('observed coverage'); a.set_xlim(0.45, 1.0); a.set_ylim(0.0, 1.02)
    a.legend(fontsize=5.0, loc='upper center', bbox_to_anchor=(1.3, -0.33), ncol=3, handlelength=1.4, columnspacing=0.8); a.set_title('(a) interval reliability', fontsize=6.8)
    b = axs[1]
    data = [[r[nm]['rho'] for r in e if np.isfinite(r[nm]['rho'])] for nm in ['head', 'ens', 'both']]
    bp = b.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color='0.0', lw=1.1))
    for patch, fc in zip(bp['boxes'], ['0.85', '0.6', '0.4']): patch.set_facecolor(fc); patch.set_edgecolor('0.0'); patch.set_linewidth(0.6)
    for k, d_ in enumerate(data): b.text(k + 1, 1.06, f'{np.mean(d_):.2f}', ha='center', fontsize=5.6)
    b.set_xticks([1, 2, 3]); b.set_xticklabels(['head', 'ensemble', 'both'], fontsize=6.2)
    b.set_ylim(-0.75, 1.15); b.axhline(0, color='0.6', lw=0.6, ls=':')
    b.set_ylabel('Spearman $\\rho$ per item'); b.set_title('(b) band reliability ranking', fontsize=6.8)
    plt.savefig(f'{FIG}/fig6_uncertainty.png')

def fig7():
    """Routing: mean gate weight per expert and corruption case; gate of the column expert vs. stripe statistic."""
    rt = json.load(open(f'{R}/routing_WDPS.json'))
    names = ['spatial 1×3×3', 'column 1×1×7', 'spectral 5×1×1', 'dilated 1×3×3']
    M = np.array([[np.mean([r['gates'][k] for r in rt if r['case'] == c]) for k in range(4)] for c in CASES])
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
    a = axs[0]; im = a.imshow(M.T, cmap='Greys', vmin=0, vmax=max(0.6, M.max()), aspect='auto')
    for i in range(M.shape[0]):
        for k in range(4): a.text(i, k, f'{M[i, k]:.2f}', ha='center', va='center', fontsize=5.0, color='white' if M[i, k] > 0.35 else 'black')
    a.set_xticks(range(6)); a.set_xticklabels(CASES, fontsize=6); a.set_yticks(range(4)); a.set_yticklabels(names, fontsize=5.6)
    a.grid(False); a.set_title('(a) mean routing weight', fontsize=6.8)
    b = axs[1]
    st = np.concatenate([np.array(r['stats']) for r in rt]); g = np.concatenate([np.array(r['gates_band']) for r in rt])
    imp = st[:, 3]; edges = [0, 0.005, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
    xc = [0.0025, 0.0275, 0.075, 0.125, 0.175, 0.225, 0.28]
    for k, (nm, mk, col) in enumerate(zip(['spatial', 'column', 'spectral', 'dilated'], ['o', 's', '^', 'D'], ['0.6', '0.35', '0.0', '0.75'])):
        sel = [np.mean(g[(imp >= lo) & (imp < hi), k] > 0) for lo, hi in zip(edges[:-1], edges[1:])]
        b.plot(xc, sel, marker=mk, color=col, ms=3.6, lw=1.0, label=nm)
    b.set_xlabel('impulse statistic $\\hat\\iota_b$'); b.set_ylabel('fraction of bands routed'); b.set_ylim(-0.03, 1.42); b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_title('(b) expert use vs. impulses', fontsize=6.8)
    b.legend(fontsize=5.0, loc='upper center', ncol=2, handlelength=1.4, columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
    plt.savefig(f'{FIG}/fig7_routing.png')

CASES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

def fig8():
    """Efficiency: held-out MPSNR vs. operations (B=103) and latency."""
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    eff = json.load(open(f'{R}/efficiency.json'))['rows']; S = json.load(open(f'{R}/summary.json'))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw=dict(wspace=0.32), sharey=True)
    off = {'WDPS': (-6, 5, 'right'), 'SSMBase': (-7, -2, 'right'), 'PromptIR3D': (0, 7, 'center'), 'QRNN3D': (0, -11, 'center'),
           'SSTLite': (6, -1, 'left'), 'SqueezeHSI': (6, -1, 'left')}
    for m in DEEP:
        e = [r for r in eff if r['model'] == m and r['B'] == 103][0]; v = np.mean(S['heldout_mean'][m])
        col, mk = STY[m]
        for a, xv in zip(axs, [e['gmacs'], e['ms']]):
            a.scatter(xv, v, s=10 + e['params'] / 4000, marker=mk, color=col if m != 'WDPS' else '0.0', edgecolor='0.0', lw=0.4, zorder=3)
            dx, dy, ha = off.get(m, (3, 3, 'left'))
            a.annotate(SHORT[m], (xv, v), xytext=(dx, dy), textcoords='offset points', fontsize=5.0, ha=ha, va='center')
    for a, ticks in zip(axs, [[1, 3, 10, 30], [200, 500, 1000, 2000]]):
        a.set_xscale('log'); a.xaxis.set_major_locator(FixedLocator(ticks)); a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    axs[0].set_xlim(0.9, 60); axs[1].set_xlim(170, 3000); axs[0].set_ylim(26.8, 31.8)
    axs[0].set_xlabel('GMAC per $64\\times64\\times103$ cube'); axs[1].set_xlabel('CPU latency (ms, 2 threads)')
    axs[0].set_ylabel('held-out MPSNR (dB)')
    axs[0].set_title('(a) operations', fontsize=6.8); axs[1].set_title('(b) latency', fontsize=6.8)
    plt.savefig(f'{FIG}/fig8_efficiency.png')

def fig9():
    """Real sensor noise: Indian Pines (220 bands, AVIRIS) and EnMAP L2A water-vapour edge bands."""
    torch.set_num_threads(2)
    from downstream import predict_tiled
    import classical as CL
    ip, en = D.load('indianpines220'), D.load('enmap')
    sel = [('Indian Pines', ip, 'ROSIS', [103]), ('EnMAP', en, 'Hyperion', [135])]
    meth = ['FastHyDe', 'QRNN3D', 'SSMBase', 'WDPS']
    fig, axs = plt.subplots(2, len(meth) + 1, figsize=(7.0, 3.05), gridspec_kw=dict(wspace=0.04, hspace=0.10))
    r = 0
    for nm, d, fold, bands in sel:
        y, wl = d['x'], d['wl']
        if nm == 'EnMAP': y = y[0:128, 32:160]  # fully valid 128x128 window (no nodata pixels)
        outs = {}; cache = f'{FIG}/.fig9_{nm[:2]}.npz'
        if os.path.exists(cache): outs = dict(np.load(cache))
        for m in meth:
            if m in outs: continue
            if m == 'FastHyDe': outs[m] = np.clip(CL.METHODS[m](y), 0, 1)
            else:
                mod = E.load_model(f"./ckpt/{m}_{fold}_s0.pt")[0]; outs[m] = np.clip(predict_tiled(mod, y, wl), 0, 1)
        np.savez_compressed(cache, **outs)
        for b in bands:
            ims = [('Observed', np.clip(y, 0, 1))] + [(SHORT[m], outs[m]) for m in meth]
            for j, (t, im) in enumerate(ims):
                lo, hi = np.percentile(im[:, :, b], [1, 99])
                a = axs[r, j]; a.imshow(im[:, :, b], cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6)); a.set_xticks([]); a.set_yticks([])
                if r == 0: a.set_title(t, fontsize=7)
            axs[r, 0].set_ylabel(f'{nm}\nband {b + 1}, {wl[b]:.0f} nm', fontsize=6.3)
            r += 1
    plt.savefig(f'{FIG}/fig9_realnoise.png')

if __name__ == '__main__':
    for f in sys.argv[1:]: globals()[f]()
