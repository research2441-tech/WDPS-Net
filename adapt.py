"""Test-time self-supervised adaptation (SSA) to an unseen sensor using ONLY its noisy cubes.
Loss: masked-band reconstruction of the observed (noisy) values + unmixing consistency with endmembers taken from the
detached current estimate + routing balance. No clean target images are used.
Usage: python adapt.py ckpt.pt [steps]"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
import wdps_data as D, train as T, evaluate as E
from wdps_metrics import all_metrics

def target_crops(fold, rng, n, P=24, Bt=24):
    items = [it for it in E.INDEX if it['cube'] in D.FOLDS[fold]]
    ys, wls = [], []
    for _ in range(n):
        it = items[rng.integers(len(items))]; d = np.load(f"{E.TS}/{it['file']}"); y, wl = d['noisy'], d['wl']
        H, W, B = y.shape; b = min(Bt, B); b0 = rng.integers(0, B - b + 1); i, j = rng.integers(0, H - P + 1, 2)
        ys.append(y[i:i + P, j:j + P, b0:b0 + b]); wls.append(wl[b0:b0 + b])
    b = min(a.shape[2] for a in ys); ys = [a[:, :, :b] for a in ys]; wls = [w[:b] for w in wls]
    return torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2).float(), torch.from_numpy(np.stack(wls)).float()

def ssa(path, steps=150, lr=3e-4, batch=6, lam_um=0.1, tag='+SSA'):
    """Pseudo-clean re-degradation: a frozen copy of the source model restores target crops; the restored crops are
    corrupted again with the training degradation model and the network learns to undo that corruption, while the
    unmixing term constrains its output on the real target observations. No clean target image is used."""
    import copy
    m, args, ck = E.load_model(path); torch.set_num_threads(int(os.environ.get('WDPS_THREADS', 2)))
    teacher = copy.deepcopy(m).eval(); m.train()
    rng = np.random.default_rng(1234 + args.seed); gen = torch.Generator().manual_seed(args.seed)
    is_wdps = args.model == 'WDPS'
    opt = torch.optim.Adam(m.parameters(), lr=lr); t0 = time.time()
    for it in range(steps):
        y, wl = target_crops(args.fold, rng, batch)
        with torch.no_grad(): xt = teacher(y, wl)['xhat'].clamp(0, 1)
        kinds = D.sample_train_kinds(rng)
        yt = torch.from_numpy(np.stack([D.degrade(xt[n].permute(1, 2, 0).numpy(), rng, kinds)[0] for n in range(len(xt))])).permute(0, 3, 1, 2)
        o = m(yt, wl)
        L = torch.sqrt((o['xhat'] - xt) ** 2 + 1e-6).mean() + (0.01 * o['bal'] if is_wdps else 0)
        if lam_um > 0:
            L = L + lam_um * T.unmix_loss(m(y, wl)['xhat'], None, gen)
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); res = []
    for itm in E.INDEX:
        if itm['cube'] not in D.FOLDS[args.fold]: continue
        d = np.load(f"{E.TS}/{itm['file']}"); xh, _ = E.predict(m, d['noisy'], d['wl'])
        res.append(dict(itm, **all_metrics(d['clean'], xh), B=int(d['clean'].shape[2])))
    name = os.path.basename(path)[:-3] + tag
    json.dump(dict(name=name, args=vars(args), adapt_steps=steps, adapt_time=time.time() - t0, items=res),
              open(f'./results/{name}.json', 'w'))
    print(name, round(np.mean([r['psnr'] for r in res]), 3), round(time.time() - t0), flush=True)

if __name__ == '__main__':
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    ssa(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150, lam_um=lam, tag='+SSA' if lam > 0 else '+SSA-um')
