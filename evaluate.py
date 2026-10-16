"""Evaluate a trained checkpoint on its held-out-sensor test items and the cross-scene items."""
import sys, json, os, time, glob, numpy as np, torch
import wdps_data as D
from wdps_metrics import all_metrics
import train as T

TS = './testsets'; INDEX = json.load(open(f'{TS}/index.json'))

def items_for(fold):
    return [it for it in INDEX if it['cube'] in D.FOLDS[fold] or it['cube'] in D.XSCENE]

def load_model(path):
    ck = torch.load(path, weights_only=False); args = T.get_args([]); args.gap_learned = 0; args.__dict__.update(ck['args'])
    m = T.build_model(args); st = dict(ck['state'])
    if 'head.weight' in st and st['head.weight'].shape[0] == 2 and hasattr(m, 'vhead'):   # checkpoints with a joint 2-channel head
        w, b = st.pop('head.weight'), st.pop('head.bias')
        st.update({'head.weight': w[:1], 'head.bias': b[:1], 'vhead.weight': w[1:], 'vhead.bias': b[1:]})
    m.load_state_dict(st); m.eval(); return m, args, ck

@torch.no_grad()
def predict(m, y, wl):
    o = m(torch.from_numpy(y).permute(2, 0, 1)[None].float(), torch.from_numpy(wl)[None].float())
    xh = o['xhat'][0].permute(1, 2, 0).numpy()
    lv = o['logv'][0].permute(1, 2, 0).numpy() if 'logv' in o else None
    return xh, lv

def eval_ckpt(path, out_dir='./results'):
    os.makedirs(out_dir, exist_ok=True)
    m, args, ck = load_model(path); torch.set_num_threads(int(os.environ.get('WDPS_THREADS', 2)))
    res = []
    for it in items_for(args.fold):
        d = np.load(f"{TS}/{it['file']}"); t = time.time()
        xh, lv = predict(m, d['noisy'], d['wl'])
        r = dict(it, **all_metrics(d['clean'], xh), sec=time.time() - t, B=int(d['clean'].shape[2]))
        if lv is not None and args.model == 'WDPS':
            err = np.clip(xh, 0, 1) - d['clean']; sig = np.sqrt(np.exp(lv))
            r['band_sigma'] = np.sqrt((sig ** 2).reshape(-1, sig.shape[2]).mean(0)).tolist()
            r['band_rmse'] = np.sqrt((err ** 2).reshape(-1, err.shape[2]).mean(0)).tolist()
            zz = np.abs(err / sig).ravel()
            from scipy.stats import norm
            r['coverage'] = {str(p): float((zz <= norm.ppf(0.5 + p / 2)).mean()) for p in [0.5, 0.8, 0.9, 0.95]}
            r['nll'] = float((0.5 * (err ** 2 / sig ** 2) + np.log(sig) + 0.5 * np.log(2 * np.pi)).mean())
        res.append(r)
    name = os.path.basename(path)[:-3]
    json.dump(dict(name=name, args=vars(args), params=ck['params'], train_time=ck['train_time'], items=res), open(f'{out_dir}/{name}.json', 'w'))
    return res

if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = eval_ckpt(p)
        import collections; agg = collections.defaultdict(list)
        for x in r: agg[x['case'][:2] + ('X' if x['cube'] in D.XSCENE else 'H')].append(x['psnr'])
        print(os.path.basename(p), {k: round(np.mean(v), 2) for k, v in sorted(agg.items())})
