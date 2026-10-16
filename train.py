"""Train WDPS-Net (and ablations) or a deep baseline on one leave-one-sensor-out fold.
Usage: python train.py --model WDPS --fold ROSIS --seed 0 --steps 1200 [ablation flags]"""
import argparse, json, math, os, time, numpy as np, torch, torch.nn.functional as F
import wdps_data as D
from wdps_models import WDPSNet, count_params
from baselines import BASELINES

def get_args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--model', default='WDPS'); a.add_argument('--fold', default='ROSIS'); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--steps', type=int, default=1200); a.add_argument('--batch', type=int, default=6); a.add_argument('--patch', type=int, default=24)
    a.add_argument('--lr', type=float, default=2e-3); a.add_argument('--tag', default='')
    a.add_argument('--wave', default='phys'); a.add_argument('--no_prompt', action='store_true'); a.add_argument('--no_route', action='store_true')
    a.add_argument('--no_spec', action='store_true'); a.add_argument('--no_spat', action='store_true'); a.add_argument('--no_gap', action='store_true'); a.add_argument('--no_subspace', action='store_true'); a.add_argument('--proj', default='raw'); a.add_argument('--gap_learned', type=int, default=1)
    a.add_argument('--K', type=int, default=4); a.add_argument('--freqs', type=int, default=8)
    a.add_argument('--lam_um', type=float, default=0.0); a.add_argument('--lam_mask', type=float, default=0.0); a.add_argument('--mask_ratio', type=float, default=0.15)
    a.add_argument('--mask_prob', type=float, default=0.0); a.add_argument('--C', type=int, default=16)
    a.add_argument('--lam_ctr', type=float, default=0.0); a.add_argument('--lam_nll', type=float, default=0.05)
    a.add_argument('--out', default='./ckpt')
    return a.parse_args(argv)

def build_model(args):
    import wdps_models as _wm
    _wm.GAP_LEARNED = bool(getattr(args, 'gap_learned', 0)) and args.model == 'WDPS'
    if args.model == 'WDPS':
        return WDPSNet(C=args.C, K=args.K, freqs=args.freqs, wave=args.wave, prompt=not args.no_prompt, routed=not args.no_route,
                       spec=not args.no_spec, spat=not args.no_spat, gap=not args.no_gap, subspace=(False if args.no_subspace else getattr(args, 'proj', 'raw')))
    return BASELINES[args.model]()

# ------------------------------------------------------------------ unmixing utilities (Eq. 13-14)
def vca(X, R, gen):
    """Vertex component analysis on pixels X:(P,B) -> endmembers (R,B)."""
    Xc = X - X.mean(0, keepdim=True)
    U = torch.linalg.svd(Xc.T @ Xc / X.shape[0])[0][:, :R]
    Xp = X @ U; Aproj = torch.zeros(R, R); idx = []
    for i in range(R):
        w = torch.randn(R, generator=gen)
        f = w - Aproj @ torch.linalg.pinv(Aproj) @ w; f = f / (f.norm() + 1e-9)
        j = int((Xp @ f).abs().argmax()); idx.append(j); Aproj[:, i] = Xp[j]
    return X[idx]

def abundances(X, E, iters=15):
    """Non-negative abundances by projected gradient: min_A ||X - A E||^2, A >= 0. X:(P,B), E:(R,B)."""
    G = E @ E.T; eta = 1.0 / (torch.linalg.matrix_norm(G, 2) + 1e-6)
    A = (X @ E.T @ torch.linalg.pinv(G)).clamp(min=0)
    for _ in range(iters):
        A = (A - eta * (A @ G - X @ E.T)).clamp(min=0)
    return A

def unmix_loss(xhat, x_ref, gen, R=5):
    """xhat,x_ref:(N,B,H,W). If x_ref is None, endmembers come from the detached estimate (target-domain mode)."""
    tot = 0
    for n in range(xhat.shape[0]):
        Xh = xhat[n].flatten(1).T
        src = (x_ref[n] if x_ref is not None else xhat[n].detach()).flatten(1).T
        with torch.no_grad(): E = vca(src, R, gen)
        Ah = abundances(Xh, E)
        l = ((Xh - Ah @ E) ** 2).sum() / (Xh ** 2).sum().clamp(min=1e-6)
        if x_ref is not None:
            with torch.no_grad(): A = abundances(src, E)
            l = l + 0.1 * (Ah - A).abs().mean()
        tot = tot + l
    return tot / xhat.shape[0]

def sam_loss(a, b):
    c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)).mean()

def info_nce(z, M, tau=0.2):
    sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(M, 2 * M), torch.arange(0, M)])
    return F.cross_entropy(sim, tgt)

# ------------------------------------------------------------------ batch sampler
class Sampler:
    def __init__(s, pool, patch, rng):
        s.rng, s.patch = rng, patch
        s.cubes = {n: D.load(n) for n in pool}
        s.by_sensor = {}
        for n in pool: s.by_sensor.setdefault(D.META[n]['sensor'], []).append(n)
    def crop(s, name, Bt):
        d = s.cubes[name]; x, wl = d['x'], d['wl']; H, W, B = x.shape; P = s.patch; rng = s.rng
        stride = 2 if (B >= 2 * Bt and rng.random() < 0.5) else 1
        b0 = rng.integers(0, B - stride * (Bt - 1)); bi = b0 + stride * np.arange(Bt)
        for _ in range(50):
            i, j = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
            c = x[i:i + P, j:j + P][:, :, bi]
            if (c.mean(2) > 1e-3).mean() > 0.95: break
        k = rng.integers(4); c = np.rot90(c, k).copy()
        if rng.random() < 0.5: c = c[::-1].copy()
        return c, wl[bi].astype(np.float32)
    def batch(s, n_pairs, Bt):
        sensors = [k for k, v in s.by_sensor.items() if max(s.cubes[c]['x'].shape[2] for c in v) >= Bt]
        cl, ny, wls = [None] * (2 * n_pairs), [None] * (2 * n_pairs), [None] * (2 * n_pairs)
        for m in range(n_pairs):
            kinds = D.sample_train_kinds(s.rng); params = None
            for v in range(2):
                sen = sensors[s.rng.integers(len(sensors))]
                names = [c for c in s.by_sensor[sen] if s.cubes[c]['x'].shape[2] >= Bt]
                w = np.array([np.prod(s.cubes[c]['x'].shape[:2]) for c in names], float) ** 0.5
                name = names[s.rng.choice(len(names), p=w / w.sum())]
                c, wl = s.crop(name, Bt)
                y, params = D.degrade(c, s.rng, kinds, params)
                cl[m + v * n_pairs], ny[m + v * n_pairs], wls[m + v * n_pairs] = c, y, wl
        t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() if a[0].ndim == 3 else torch.from_numpy(np.stack(a)).float()
        return t(cl), t(ny), t(wls)

def compute_loss(model, args, x, y, wl, gen, rng, is_wdps):
    if not is_wdps:
        o = model(y, wl); return F.l1_loss(o['xhat'], x) + 0.1 * sam_loss(o['xhat'], x), {}
    N, B = y.shape[:2]
    mask = torch.zeros(N, B)
    if args.mask_prob > 0:
        for n in range(N):
            if rng.random() < args.mask_prob:
                k = max(1, int(round(args.mask_ratio * B))); mask[n, torch.from_numpy(rng.choice(B, k, replace=False))] = 1
    o = model(y, wl, mask if mask.sum() > 0 else None)
    xh = o['xhat']
    char = torch.sqrt((xh - x) ** 2 + 1e-6).mean()
    L = char + 0.1 * sam_loss(xh, x); parts = dict(char=char.item())
    nll = 0.5 * (torch.exp(-o['logv']) * (xh.detach() - x) ** 2 + o['logv']).mean()
    L = L + args.lam_nll * nll
    if args.lam_mask > 0 and mask.sum() > 0:
        mm = mask[..., None, None].expand_as(y)
        lm = (((xh - y) ** 2) * mm).sum() / mm.sum()
        L = L + args.lam_mask * lm; parts['mask'] = lm.item()
    if args.lam_um > 0:
        lu = unmix_loss(xh, x, gen); L = L + args.lam_um * lu; parts['um'] = lu.item()
    if args.lam_ctr > 0 and o['z'] is not None:
        lc = info_nce(o['z'], N // 2); L = L + args.lam_ctr * lc; parts['ctr'] = lc.item()
    L = L + 0.01 * o['bal']
    return L, parts

def run_name(args):
    return f"{args.model}{args.tag}_{args.fold.replace('/', '-')}_s{args.seed}"

def main(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); gen = torch.Generator().manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('WDPS_THREADS', 2)))
    model = build_model(args); is_wdps = args.model == 'WDPS'
    if not is_wdps: args.lam_mask = 0; args.mask_prob = 0
    sampler = Sampler(D.train_pool(args.fold), args.patch, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = lambda it: min(1, (it + 1) / 50) * (0.5 * (1 + math.cos(math.pi * it / args.steps)) * 0.99 + 0.01)
    os.makedirs(args.out, exist_ok=True); name = run_name(args); log = []; t0 = time.time()
    for it in range(args.steps):
        for g in opt.param_groups: g['lr'] = args.lr * sched(it)
        Bt = int(rng.choice([16, 24, 24, 32]))
        x, y, wl = sampler.batch(args.batch // 2, Bt)
        L, parts = compute_loss(model, args, x, y, wl, gen, rng, is_wdps)
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == args.steps - 1:
            log.append(dict(it=it, loss=L.item(), t=time.time() - t0, **parts))
            print(name, it, round(L.item(), 4), parts, round(time.time() - t0), flush=True)
    torch.save(dict(state=model.state_dict(), args=vars(args), params=count_params(model), train_time=time.time() - t0, log=log), f'{args.out}/{name}.pt')

if __name__ == '__main__':
    main(get_args())
