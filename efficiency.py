"""Parameters, operation counts, latency and peak memory of every learned model on 64x64xB cubes (B = 31, 103, 204).
Operations: multiply-accumulates of convolutions and matrix products counted by torch FlopCounterMode, plus the
element-wise operations of the selective scans counted analytically (5 per state element per step)."""
import sys, os, json, time, resource, subprocess, numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
import train as T, wdps_models as WM

def build(name):
    a = T.get_args(['--model', name]); return T.build_model(a)

def scan_ops(model, y, wl):
    tot = [0]
    def hook(mod, inp, out):
        x = inp[0]; Nn, L, Dd = x.shape; S = mod.A_log.shape[1]; tot[0] += 5 * Nn * L * Dd * S
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, WM.SelectiveSSM)]
    with torch.no_grad(): model(y, wl)
    for h in hs: h.remove()
    return tot[0]

def attn_ops(model, B, H=64, W=64, N=1):
    """MACs of nn.MultiheadAttention, whose fused inference kernel is invisible to FlopCounterMode:
    input/output projections (4 C^2 per token) plus QK^T and AV products for the spatial windows and the spectral axis."""
    if not hasattr(model, 'blocks'): return 0
    tot = 0
    for bk in model.blocks:
        if 'sa' not in bk: continue
        C = bk['sa'].embed_dim; tok = N * B * H * W; w = model.win
        tot += 2 * 4 * C * C * tok                       # projections of the spatial and spectral attention
        tot += 2 * tok * (w * w) * C                     # windowed spatial attention (QK^T and AV)
        tot += 2 * tok * B * C                           # global spectral attention over B bands (QK^T and AV)
    return tot

def one(name, B, threads):
    torch.set_num_threads(threads); torch.manual_seed(0)
    m = build(name).eval(); y = torch.rand(1, B, 64, 64); wl = torch.linspace(400, 2400, B)[None]
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc: m(y, wl)
        flops = fc.get_total_flops() / 2 + scan_ops(m, y, wl) + attn_ops(m, B)
        m(y, wl); ts = []
        for _ in range(5):
            t = time.perf_counter(); m(y, wl); ts.append(time.perf_counter() - t)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return dict(model=name, B=B, params=sum(p.numel() for p in m.parameters()), gmacs=flops / 1e9, ms=1000 * float(np.median(ts)), peak_mb=peak)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'one':
        print(json.dumps(one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))); sys.exit()
    rows = []
    for name in ['WDPS', 'SSMBase', 'QRNN3D', 'SSTLite', 'PromptIR3D', 'SqueezeHSI']:
        for B in [31, 103, 204]:
            r = subprocess.run([sys.executable, __file__, 'one', name, str(B), '2'], capture_output=True, text=True, cwd='.')
            rows.append(json.loads(r.stdout.strip().splitlines()[-1])); print(rows[-1], flush=True)
    base = subprocess.run([sys.executable, '-c', 'import resource,torch,train;print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)'],
                          capture_output=True, text=True, cwd='.').stdout.strip()
    json.dump(dict(rows=rows, import_baseline_mb=float(base)), open('./results/efficiency.json', 'w'), indent=1)
