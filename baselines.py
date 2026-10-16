"""Compact, band-agnostic re-implementations of deep HSI denoising baselines (same I/O as WDPS-Net:
forward(y:(N,B,H,W), wl:(N,B)) -> dict(xhat=...)). Capacities are matched to roughly 15-60k parameters."""
import torch, torch.nn as nn, torch.nn.functional as F
from wdps_models import scan, WDPSNet

class Fire(nn.Module):
    def __init__(s, cin, sq, ex):
        super().__init__(); s.sq = nn.Conv2d(cin, sq, 1); s.e1 = nn.Conv2d(sq, ex, 1); s.e3 = nn.Conv2d(sq, ex, 3, padding=1)
    def forward(s, x):
        z = F.relu(s.sq(x)); return torch.cat([F.relu(s.e1(z)), F.relu(s.e3(z))], 1)

class SqueezeHSI(nn.Module):
    """Direct predecessor: SqueezeNet fire modules on a sliding window of k adjacent bands (band-agnostic adaptation)."""
    def __init__(s, k=7, C=32):
        super().__init__(); s.k = k
        s.net = nn.Sequential(nn.Conv2d(k, C, 3, padding=1), nn.ReLU(), Fire(C, 8, C // 2), Fire(C, 8, C // 2), Fire(C, 8, C // 2), nn.Conv2d(C, 1, 3, padding=1))
    def forward(s, y, wl):
        N, B, H, W = y.shape; r = s.k // 2
        yp = torch.cat([y[:, :1].expand(-1, r, -1, -1), y, y[:, -1:].expand(-1, r, -1, -1)], 1)   # (N,B+2r,H,W)
        win = yp.unfold(1, s.k, 1).permute(0, 1, 4, 2, 3).reshape(N * B, s.k, H, W)
        return dict(xhat=y - s.net(win).reshape(N, B, H, W))

class QRNN3D(nn.Module):
    """3D conv encoder-decoder with alternating-direction quasi-recurrent pooling along the spectrum."""
    class QL(nn.Module):
        def __init__(s, cin, C, rev, stride=(1, 1, 1), up=False):
            super().__init__(); s.rev = rev
            s.conv = (nn.ConvTranspose3d(cin, 2 * C, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)) if up
                      else nn.Conv3d(cin, 2 * C, 3, stride=stride, padding=1))
        def forward(s, x):
            z, f = s.conv(x).chunk(2, 1); z, f = torch.tanh(z), torch.sigmoid(f)
            z, f = z.permute(0, 2, 1, 3, 4), f.permute(0, 2, 1, 3, 4)          # (N,B,C,H,W)
            if s.rev: z, f = z.flip(1), f.flip(1)
            h = scan(f, (1 - f) * z)
            if s.rev: h = h.flip(1)
            return h.permute(0, 2, 1, 3, 4)
    def __init__(s, C=16):
        super().__init__()
        s.e1 = s.QL(1, C, False); s.e2 = s.QL(C, 2 * C, True, (1, 2, 2)); s.m = s.QL(2 * C, 2 * C, False)
        s.d2 = s.QL(2 * C, C, True, up=True); s.d1 = s.QL(C, C, False); s.out = nn.Conv3d(C, 1, 3, padding=1)
    def forward(s, y, wl):
        x = y[:, None]; a = s.e1(x); b = s.e2(a); b = b + s.m(b); c = s.d2(b)[..., :a.shape[-2], :a.shape[-1]] + a
        return dict(xhat=y - s.out(s.d1(c))[:, 0])

class SSTLite(nn.Module):
    """Spatial-spectral Transformer: windowed spatial self-attention per band + global spectral self-attention per pixel."""
    def __init__(s, C=16, heads=2, win=8, nblk=2):
        super().__init__(); s.win = win
        s.stem = nn.Conv3d(1, C, 3, padding=1); s.out = nn.Conv3d(C, 1, 3, padding=1)
        s.blocks = nn.ModuleList([nn.ModuleDict(dict(n1=nn.LayerNorm(C), sa=nn.MultiheadAttention(C, heads, batch_first=True),
                                                     n2=nn.LayerNorm(C), se=nn.MultiheadAttention(C, heads, batch_first=True),
                                                     n3=nn.LayerNorm(C), ff=nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))))
                                     for _ in range(nblk)])
    def forward(s, y, wl):
        N, B, H, W = y.shape; w = s.win
        x = s.stem(y[:, None]).permute(0, 2, 3, 4, 1)                          # (N,B,H,W,C)
        ph, pw = (-H) % w, (-W) % w
        for bk in s.blocks:
            t = F.pad(bk['n1'](x), (0, 0, 0, pw, 0, ph)); Hp, Wp = H + ph, W + pw; C = t.shape[-1]
            t = t.reshape(N * B, Hp // w, w, Wp // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)
            t = bk['sa'](t, t, t, need_weights=False)[0]
            t = t.reshape(N * B, Hp // w, Wp // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(N, B, Hp, Wp, C)[:, :, :H, :W]
            x = x + t
            t = bk['n2'](x).permute(0, 2, 3, 1, 4).reshape(N * H * W, B, C)
            t = torch.cat([bk['se'](u, u, u, need_weights=False)[0] for u in t.split(1024)], 0).reshape(N, H, W, B, C).permute(0, 3, 1, 2, 4)
            x = x + t
            x = x + bk['ff'](bk['n3'](x))
        return dict(xhat=y - s.out(x.permute(0, 4, 1, 2, 3))[:, 0])

class SMDSLite(nn.Module):
    """Model-guided unrolled convolutional sparse coding (LISTA) with 3D spectral-spatial dictionaries."""
    def __init__(s, C=16, iters=5):
        super().__init__(); s.iters = iters
        s.We = nn.Conv3d(1, C, 3, padding=1); s.Wd = nn.ModuleList([nn.Conv3d(C, 1, 3, padding=1) for _ in range(iters)])
        s.Wt = nn.ModuleList([nn.Conv3d(1, C, 3, padding=1) for _ in range(iters)])
        s.theta = nn.Parameter(torch.full((iters, C), 0.02)); s.dec = nn.Conv3d(C, 1, 3, padding=1)
    def forward(s, y, wl):
        x = y[:, None]; z = self_soft(s.We(x), s.theta[0])
        for k in range(s.iters):
            z = self_soft(z - s.Wt[k](s.Wd[k](z) - x), s.theta[k])
        return dict(xhat=s.dec(z)[:, 0])

def self_soft(z, th):
    th = th.abs()[None, :, None, None, None]; return torch.sign(z) * F.relu(z.abs() - th)

class PromptIR3D(nn.Module):
    """Prompt-based all-in-one restoration: learnable prompt components weighted by global features (PromptIR mechanism, 3D)."""
    class RB(nn.Module):
        def __init__(s, C):
            super().__init__(); s.c1 = nn.Conv3d(C, C, 3, padding=1); s.c2 = nn.Conv3d(C, C, 3, padding=1)
        def forward(s, x): return x + s.c2(F.gelu(s.c1(x)))
    class PGB(nn.Module):
        def __init__(s, C, n=5, Cp=8):
            super().__init__(); s.comp = nn.Parameter(torch.randn(n, Cp, 4, 4, 4) * 0.1); s.w = nn.Linear(C, n); s.fuse = nn.Conv3d(C + Cp, C, 3, padding=1)
        def forward(s, x):
            a = torch.softmax(s.w(x.mean((2, 3, 4))), -1)
            p = torch.einsum('nk,kcdhw->ncdhw', a, s.comp)
            p = F.interpolate(p, size=x.shape[2:], mode='trilinear', align_corners=False)
            return x + s.fuse(torch.cat([x, p], 1))
    def __init__(s, C=16):
        super().__init__()
        s.stem = nn.Conv3d(1, C, 3, padding=1); s.down = nn.Conv3d(C, C, 3, stride=(1, 2, 2), padding=1)
        s.b1, s.b2 = s.RB(C), s.RB(C); s.p1, s.p2 = s.PGB(C), s.PGB(C)
        s.up = nn.ConvTranspose3d(C, C, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)); s.b3 = s.RB(C); s.out = nn.Conv3d(C, 1, 3, padding=1)
    def forward(s, y, wl):
        a = s.stem(y[:, None]); b = s.p1(s.b1(s.down(a))); b = s.p2(s.b2(b))
        c = s.up(b)[..., :a.shape[-2], :a.shape[-1]] + a
        return dict(xhat=y - s.out(s.b3(c))[:, 0])

def SSMBase():
    """Spatial-spectral selective-SSM baseline (SSUMamba-style design, factorized bidirectional scans), without
    wavelength conditioning, degradation prompting, routing or gap-aware discretization."""
    return WDPSNet(C=16, wave='index', prompt=False, routed=False, gap=False, subspace=False)

BASELINES = dict(SqueezeHSI=SqueezeHSI, QRNN3D=QRNN3D, SSTLite=SSTLite, SMDSLite=SMDSLite, PromptIR3D=PromptIR3D, SSMBase=SSMBase)
