"""WDPS-Net: Wavelength-Conditioned Degradation-Prompted State-Space Network (CPU reference implementation).
Tensor layout throughout: (N, C, B, H, W) with B spectral bands; wavelengths wl: (N, B) in nm."""
import math, torch, torch.nn as nn, torch.nn.functional as F

LAM0, LAM_SPAN, DLAM_REF = 350.0, 2200.0, 10.0

class ChanLN(nn.Module):
    def __init__(s, C):
        super().__init__(); s.ln = nn.LayerNorm(C)
    def forward(s, x):
        return s.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

# ---------------------------------------------------------------- wavelength embedding (Eq. 1-2)
class WaveEmbed(nn.Module):
    def __init__(s, C, K=8, mode='phys'):
        super().__init__(); s.mode = mode
        s.register_buffer('freq', (2.0 ** torch.arange(K)) * math.pi)
        s.mlp = nn.Sequential(nn.Linear(2 * K, 32), nn.GELU(), nn.Linear(32, C))
    def forward(s, wl):
        if s.mode == 'phys':
            t = (wl - LAM0) / LAM_SPAN
        else:   # ablation: relative band index only (no physical wavelength)
            B = wl.shape[1]; t = torch.linspace(0, 1, B, device=wl.device)[None].expand_as(wl)
        a = t[..., None] * s.freq
        return s.mlp(torch.cat([torch.sin(a), torch.cos(a)], -1))           # (N,B,C)

# ---------------------------------------------------------------- degradation descriptors (Eq. 5)
def _median3x3(y):
    N, B, H, W = y.shape
    u = F.unfold(F.pad(y.reshape(N * B, 1, H, W), (1, 1, 1, 1), mode='replicate'), 3)
    return u.median(1).values.reshape(N, B, H, W)

def degradation_stats(y):
    """y: (N,B,H,W) observed cube. Returns (N,B,5) clean-free descriptors."""
    N, B, H, W = y.shape
    d = (y[:, :, 0:H - H % 2:2, 0:W - W % 2:2] - y[:, :, 1::2, 0:W - W % 2:2] - y[:, :, 0:H - H % 2:2, 1::2] + y[:, :, 1::2, 1::2]) / 2
    sig = d.abs().flatten(2).median(-1).values / 0.6745
    cm = y.mean(2)                                                     # column means (N,B,W)
    det = cm - F.avg_pool1d(F.pad(cm.reshape(N * B, 1, W), (3, 3), mode='replicate'), 7, 1).reshape(N, B, W)
    stripe = det.abs().mean(-1)
    dead = (y.std(2) < 1e-3).float().mean(-1)
    imp = ((y - _median3x3(y)).abs() > 0.3).float().flatten(2).mean(-1)
    yp = F.pad(y.permute(0, 2, 3, 1).reshape(-1, 1, B), (1, 1), mode='replicate').reshape(N, H, W, B + 2)
    rough = (yp[..., 1:-1] - 0.5 * (yp[..., :-2] + yp[..., 2:])).abs().flatten(1, 2).median(1).values
    return torch.stack([torch.log(sig + 1e-3), torch.log(stripe + 1e-3), dead, imp, torch.log(rough + 1e-3)], -1)

class DegradationEncoder(nn.Module):
    def __init__(s, P=16, Ce=16):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.GELU())
        s.mlp = nn.Sequential(nn.Linear(5 + 16 + Ce, 32), nn.GELU(), nn.Linear(32, P))
        s.proj = nn.Sequential(nn.Linear(P, 32), nn.GELU(), nn.Linear(32, 16))   # projection head for InfoNCE
    def forward(s, y, e):
        N, B, H, W = y.shape
        st = degradation_stats(y)
        hp = y - F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), 3, 1)
        f = s.cnn(hp.reshape(N * B, 1, H, W)).flatten(2)
        f = torch.cat([f.abs().mean(-1), f.std(-1)], -1).reshape(N, B, 16)
        p = s.mlp(torch.cat([st, f, e], -1))                                # (N,B,P) bandwise prompts
        z = F.normalize(s.proj(p.mean(1)), dim=-1)                           # global prompt embedding
        return p, z, st

# ---------------------------------------------------------------- unmixing-subspace prior (Eq. 3-4)
RANKS = (4, 10)
PROJ_SOURCE = 'raw'
GAP_LEARNED = False

@torch.no_grad()
def subspace_projections(y, ranks=None):
    ranks = ranks or RANKS
    """Linear-mixing prior: clean spectra lie near the low-dimensional affine span of the endmembers. The span is
    estimated from a 3x3 spatially median-filtered copy of the observation (robust to impulses, dead lines and stripes),
    and the raw observation is projected onto it. y:(N,B,H,W) -> (N,len(ranks),B,H,W)."""
    N, B, H, W = y.shape
    ym = _median3x3(y).flatten(2)                                         # (N,B,HW)
    mu = ym.mean(2, keepdim=True); Z = ym - mu
    C = Z @ Z.transpose(1, 2) / Z.shape[2]                                # (N,B,B)
    evals, evecs = torch.linalg.eigh(C.double())
    out = []
    Y = (ym if PROJ_SOURCE == 'median' else y.flatten(2)) - mu
    for r in ranks:
        U = evecs[..., -min(r, B - 1):].float()                           # (N,B,r)
        out.append((U @ (U.transpose(1, 2) @ Y) + mu).reshape(N, B, H, W))
    return torch.stack(out, 1)

# ---------------------------------------------------------------- degradation-routed experts (Eq. 6-7)
class Expert(nn.Module):
    def __init__(s, C, k, dil=1):
        super().__init__()
        pad = tuple(dil * (kk // 2) for kk in k)
        s.dw = nn.Conv3d(C, C, k, padding=pad, dilation=(1, dil, dil), groups=C); s.pw = nn.Conv3d(C, C, 1)
    def forward(s, x):
        return s.pw(F.gelu(s.dw(x)))

class RoutedExperts(nn.Module):
    """K lightweight experts; per-band top-k gating driven by the degradation prompt and the wavelength code."""
    def __init__(s, C, P, Ce, K=4, topk=2, routed=True):
        super().__init__()
        kern = [((1, 3, 3), 1), ((1, 1, 7), 1), ((5, 1, 1), 1), ((1, 3, 3), 2), ((1, 7, 1), 1), ((3, 3, 3), 1)][:K]
        s.experts = nn.ModuleList([Expert(C, k, d) for k, d in kern])
        s.router = nn.Linear(P + Ce, K); s.K, s.topk, s.routed = K, topk, routed; s.norm = ChanLN(C)
    def forward(s, x, p, e):
        if s.routed:
            logits = s.router(torch.cat([p, e], -1))                         # (N,B,K)
            v, i = logits.topk(s.topk, -1)
            g = torch.zeros_like(logits).scatter(-1, i, torch.softmax(v, -1))
            probs = torch.softmax(logits, -1)
        else:
            g = torch.full(p.shape[:2] + (s.K,), 1.0 / s.K, device=x.device); probs = g
        h = s.norm(x); out = 0
        for k, E in enumerate(s.experts):
            out = out + E(h) * g[..., k][:, None, :, None, None]
        f = g.mean((0, 1)) if not s.routed else (g > 0).float().mean((0, 1))
        bal = s.K * (f * probs.mean((0, 1))).sum()                           # switch-style load-balancing term
        return x + out, bal, g

# ---------------------------------------------------------------- selective state-space scan (Eq. 8-10)
def scan(dA, dBx, T=8):
    """Linear recurrence h_t = dA_t * h_{t-1} + dBx_t over dim 1, two-level chunked evaluation:
    T intra-chunk steps run for all chunks in parallel, then a short carry pass across chunks."""
    Nn, L = dA.shape[:2]
    if L <= T:
        a_l, b_l = dA.unbind(1), dBx.unbind(1); h = torch.zeros_like(b_l[0]); hs = []
        for t in range(L):
            h = a_l[t] * h + b_l[t]; hs.append(h)
        return torch.stack(hs, 1)
    pad = (-L) % T
    if pad:
        dA = torch.cat([dA, dA.new_ones((Nn, pad) + dA.shape[2:])], 1); dBx = torch.cat([dBx, dBx.new_zeros((Nn, pad) + dBx.shape[2:])], 1)
    nc = dA.shape[1] // T
    a = dA.reshape((Nn, nc, T) + dA.shape[2:]); b = dBx.reshape((Nn, nc, T) + dBx.shape[2:])
    a_l, b_l = a.unbind(2), b.unbind(2)
    h = torch.zeros_like(b_l[0]); p = torch.ones_like(a_l[0]); hs, ps = [], []
    for t in range(T):                       # local states from zero initial state, and cumulative decay
        h = a_l[t] * h + b_l[t]; p = p * a_l[t]; hs.append(h); ps.append(p)
    H = torch.stack(hs, 2); Pc = torch.stack(ps, 2)                     # (Nn,nc,T,...)
    ends, prods = H[:, :, -1].unbind(1), Pc[:, :, -1].unbind(1)
    carry = [torch.zeros_like(ends[0])]
    for c in range(nc - 1):
        carry.append(prods[c] * carry[-1] + ends[c])
    Cr = torch.stack(carry, 1)[:, :, None]                              # carry entering each chunk
    out = (H + Pc * Cr).reshape((Nn, nc * T) + dA.shape[2:])
    return out[:, :L]

class SelectiveSSM(nn.Module):
    """Unidirectional selective SSM: h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t,  y_t = C_t h_t + D x_t."""
    def __init__(s, D, S=4):
        super().__init__()
        s.dt = nn.Linear(D, D); s.Bp = nn.Linear(D, S); s.Cp = nn.Linear(D, S)
        s.A_log = nn.Parameter(torch.log(torch.arange(1, S + 1).float()).repeat(D, 1)); s.Dk = nn.Parameter(torch.ones(D))
        nn.init.constant_(s.dt.bias, math.log(math.e ** 0.5 - 1))
    def forward(s, x, gap=None):
        Nn, L, D = x.shape
        dt = F.softplus(s.dt(x))
        if gap is not None: dt = dt * gap[..., None]
        A = -torch.exp(s.A_log)
        dA = torch.exp(dt[..., None] * A); dBx = (dt * x)[..., None] * s.Bp(x)[:, :, None, :]
        Cm = s.Cp(x)
        hs = scan(dA, dBx)
        return (hs * Cm[:, :, None, :]).sum(-1) + x * s.Dk

class BiSSM(nn.Module):
    def __init__(s, D, S=4, Di=8):
        super().__init__(); s.inp = nn.Linear(D, 2 * Di); s.f = SelectiveSSM(Di, S); s.b = SelectiveSSM(Di, S); s.out = nn.Linear(Di, D)
    def forward(s, x, gap_f=None, gap_b=None):
        u, z = s.inp(x).chunk(2, -1)
        yf = s.f(u, gap_f)
        yb = s.b(u.flip(1), None if gap_b is None else gap_b.flip(1)).flip(1)
        return s.out((yf + yb) * F.silu(z))

def spectral_gaps(wl, clip=(0.25, 8.0)):
    import wdps_models as _m
    d = (wl[:, 1:] - wl[:, :-1]).abs() / _m.DLAM_REF
    gf = torch.cat([torch.ones_like(d[:, :1]), d], 1).clamp(*clip)       # gap entering band b from b-1
    gb = torch.cat([d, torch.ones_like(d[:, :1])], 1).clamp(*clip)       # gap entering band b from b+1 (reverse scan)
    return gf, gb

class SpectralSSM(nn.Module):
    def __init__(s, C, S=4, use_ssm=True, gap=True):
        super().__init__(); s.use_ssm, s.gap = use_ssm, gap
        s.ssm = BiSSM(C, S) if use_ssm else None
        s.conv = None if use_ssm else nn.Sequential(nn.Conv3d(C, C, (5, 1, 1), padding=(2, 0, 0), groups=C), nn.GELU(), nn.Conv3d(C, C, 1))
    def forward(s, x, e, wl):
        N, C, B, H0, W0 = x.shape
        if not s.use_ssm: return s.conv(x)
        H, W = H0, W0
        seq = (x + e.transpose(1, 2)[..., None, None]).permute(0, 3, 4, 2, 1).reshape(N * H * W, B, C)
        gf = gb = None
        if s.gap:
            gf, gb = spectral_gaps(wl)
            gf = gf.repeat_interleave(H * W, 0); gb = gb.repeat_interleave(H * W, 0)
        return s.ssm(seq, gf, gb).reshape(N, H, W, B, C).permute(0, 4, 3, 1, 2)

class SpatialSSM(nn.Module):
    """Bidirectional row + column scans at half resolution (linear in the number of pixels)."""
    def __init__(s, C, S=4, use_ssm=True):
        super().__init__(); s.use_ssm = use_ssm
        s.ssm = BiSSM(C, S) if use_ssm else None
        s.conv = None if use_ssm else nn.Sequential(nn.Conv3d(C, C, (1, 3, 3), padding=(0, 1, 1), groups=C), nn.GELU(), nn.Conv3d(C, C, 1))
    def forward(s, x):
        if not s.use_ssm: return s.conv(x)
        N, C, B, H, W = x.shape
        xd = F.avg_pool3d(x, (1, 2, 2)); h, w = xd.shape[-2:]
        r = s.ssm(xd.permute(0, 2, 3, 4, 1).reshape(N * B * h, w, C)).reshape(N, B, h, w, C)
        c = s.ssm(xd.permute(0, 2, 4, 3, 1).reshape(N * B * w, h, C)).reshape(N, B, w, h, C).transpose(2, 3)
        y = (r + c).permute(0, 4, 1, 2, 3)
        return F.interpolate(y, size=(B, H, W), mode='trilinear', align_corners=False)

class SSBlock(nn.Module):
    def __init__(s, C, S=4, spec=True, spat=True, gap=True):
        super().__init__()
        s.n1, s.n2, s.n3 = ChanLN(C), ChanLN(C), ChanLN(C)
        s.spec = SpectralSSM(C, S, spec, gap); s.spat = SpatialSSM(C, S, spat)
        s.ffn = nn.Sequential(nn.Conv3d(C, 2 * C, 1), nn.Conv3d(2 * C, 2 * C, (1, 3, 3), padding=(0, 1, 1), groups=2 * C), nn.GELU(), nn.Conv3d(2 * C, C, 1))
    def forward(s, x, e, wl):
        x = x + s.spec(s.n1(x), e, wl)
        x = x + s.spat(s.n2(x))
        return x + s.ffn(s.n3(x))

# ---------------------------------------------------------------- full network
class WDPSNetV1(nn.Module):  # first prototype (not used in the paper)
    def __init__(s, C=16, P=16, K=4, S=4, nblocks=2, wave='phys', prompt=True, routed=True, spec=True, spat=True, gap=True, freqs=8,
                 subspace=True):
        super().__init__()
        s.prompt, s.subspace = prompt, subspace
        s.we = WaveEmbed(C, freqs, wave)
        s.enc = DegradationEncoder(P, C) if prompt else None
        s.pfeat = None if prompt else nn.Linear(C, P)
        s.stem = nn.Conv3d(4 if subspace else 2, C, 3, padding=1)
        s.film = nn.Linear(C + (P if prompt else 0), 2 * C)
        s.moe1 = RoutedExperts(C, P, C, K, routed=routed); s.moe2 = RoutedExperts(C, P, C, K, routed=routed)
        s.down = nn.Conv3d(C, C, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        s.blocks = nn.ModuleList([SSBlock(C, S, spec, spat, gap) for _ in range(nblocks)])
        s.up = nn.ConvTranspose3d(C, C, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        s.head = nn.Conv3d(C, 2, 3, padding=1)
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)
    def forward(s, y, wl, mask=None):
        """y: (N,B,H,W); wl: (N,B); mask: (N,B) 1 = band hidden from the network (masked self-supervision)."""
        N, B, H, W = y.shape
        m = torch.zeros(N, B, device=y.device) if mask is None else mask
        yin = y * (1 - m)[..., None, None]
        e = s.we(wl)
        if s.prompt:
            p, z, st = s.enc(yin, e)
        else:
            z = st = None
        inp = [yin[:, None], m[:, None, :, None, None].expand(N, 1, B, H, W)]
        if s.subspace: inp.append(subspace_projections(yin))
        x = s.stem(torch.cat(inp, 1))
        if not s.prompt:
            p = s.pfeat(x.mean((3, 4)).transpose(1, 2))
        gb = s.film(torch.cat([e, p], -1) if s.prompt else e)
        gam, bet = gb.chunk(2, -1)
        x = x * (1 + gam.transpose(1, 2)[..., None, None]) + bet.transpose(1, 2)[..., None, None]
        x, bal1, g1 = s.moe1(x, p, e)
        h = s.down(x)                                                        # half-resolution latent
        for blk in s.blocks: h = blk(h, e, wl)
        x = x + s.up(h)[..., :H, :W]
        x, bal2, g2 = s.moe2(x, p, e)
        o = s.head(x)
        noise, logv = o[:, 0], o[:, 1].clamp(-12, 2) - 6.0
        xhat = yin - noise
        return dict(xhat=xhat, logv=logv, z=z, stats=st, bal=bal1 + bal2, gates=g1, prompt=p)

def count_params(m):
    return sum(p.numel() for p in m.parameters())


# =====================================================================================================================
# WDPS-Net (final): encoder-decoder whose spectral operators are wavelength-stepped selective recurrences.
# =====================================================================================================================
class WSRecurrence(nn.Module):
    """Wavelength-stepped selective recurrence along the spectrum, evaluated for every voxel and channel:
       z_b = tanh(Conv_z x)_b,  Delta_b = softplus(Conv_a x)_b * g_b,  f_b = exp(-Delta_b * a),
       h_b = f_b * h_{b-1} + (1 - f_b) * z_b   (zero-order-hold discretisation of a one-state SSM per channel).
       g_b = clip(|lambda_b - lambda_{b-1}|/delta_0) (physical gap) or 1 (index stepping)."""
    def __init__(s, cin, C, reverse=False, stride=(1, 1, 1), up=False, gap=True, recur=True):
        super().__init__(); s.reverse, s.gap, s.recur = reverse, gap, recur
        s.conv = (nn.ConvTranspose3d(cin, 2 * C, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)) if up
                  else nn.Conv3d(cin, 2 * C, 3, stride=stride, padding=1))
        s.A_log = nn.Parameter(torch.zeros(C))
        s.kappa = nn.Parameter(torch.zeros(C)) if (GAP_LEARNED and gap and recur) else None    # learned gap sensitivity, sigmoid -> (0,1)
    def forward(s, x, wl, size=None):
        u = s.conv(x)
        if size is not None: u = u[..., :size[0], :size[1]]
        z, a = u.chunk(2, 1); z = torch.tanh(z)
        if not s.recur: return z
        dt = F.softplus(a)
        if s.gap:
            gf, gb = spectral_gaps(wl); g = (gb if s.reverse else gf)[:, None, :, None, None]
            if s.kappa is not None: g = g ** torch.sigmoid(s.kappa)[None, :, None, None, None]
            dt = dt * g
        f = torch.exp(-dt * torch.exp(s.A_log)[None, :, None, None, None])
        f, z = f.permute(0, 2, 1, 3, 4), z.permute(0, 2, 1, 3, 4)               # (N,B,C,H,W)
        if s.reverse: f, z = f.flip(1), z.flip(1)
        h = scan(f, (1 - f) * z)
        if s.reverse: h = h.flip(1)
        return h.permute(0, 2, 1, 3, 4)

class WDPSNet(nn.Module):
    def __init__(s, C=16, P=16, K=4, S=4, nblocks=2, wave='phys', prompt=True, routed=True, spec=True, spat=True, gap=True, freqs=8,
                 subspace=True):
        super().__init__()
        s.prompt, s.subspace, s.spat_on = prompt, bool(subspace), spat
        s.proj_source = subspace if isinstance(subspace, str) else 'raw'
        s.we = WaveEmbed(C, freqs, wave)
        s.enc = DegradationEncoder(P, C) if prompt else None
        s.pfeat = None if prompt else nn.Linear(C, P)
        s.stem = nn.Conv3d(4 if subspace else 2, C, 3, padding=1)
        s.film = nn.Linear(C + (P if prompt else 0), 2 * C)
        s.moe1 = RoutedExperts(C, P, C, K, routed=routed); s.moe2 = RoutedExperts(C, P, C, K, routed=routed)
        kw = dict(gap=gap, recur=spec)
        s.e1 = WSRecurrence(C, C, False, **kw)
        s.e2 = WSRecurrence(C, 2 * C, True, stride=(1, 2, 2), **kw)
        s.m1 = WSRecurrence(2 * C, 2 * C, False, **kw)
        s.spat = SpatialSSM(2 * C, S, spat); s.nsp = ChanLN(2 * C)
        s.d2 = WSRecurrence(2 * C, C, True, up=True, **kw)
        s.d1 = WSRecurrence(C, C, False, **kw)
        s.head = nn.Conv3d(C, 1, 3, padding=1)                 # noise residual
        s.vhead = nn.Conv3d(C, 1, 3, padding=1)                # log-variance, fitted on detached features
        for h in (s.head, s.vhead): nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
    def forward(s, y, wl, mask=None):
        N, B, H, W = y.shape
        m = torch.zeros(N, B, device=y.device) if mask is None else mask
        yin = y * (1 - m)[..., None, None]
        e = s.we(wl)
        if s.prompt: p, z, st = s.enc(yin, e)
        else: z = st = None
        inp = [yin[:, None], m[:, None, :, None, None].expand(N, 1, B, H, W)]
        if s.subspace:
            global PROJ_SOURCE
            PROJ_SOURCE = s.proj_source; inp.append(subspace_projections(yin))
        x = s.stem(torch.cat(inp, 1))
        if not s.prompt: p = s.pfeat(x.mean((3, 4)).transpose(1, 2))
        gam, bet = s.film(torch.cat([e, p], -1) if s.prompt else e).chunk(2, -1)
        x = x * (1 + gam.transpose(1, 2)[..., None, None]) + bet.transpose(1, 2)[..., None, None]
        x, bal1, g1 = s.moe1(x, p, e)
        a = s.e1(x, wl) + x
        b = s.e2(a, wl)
        b = b + s.m1(b, wl)
        if s.spat_on: b = b + s.spat(s.nsp(b))
        c = s.d2(b, wl, size=(H, W)) + a
        c = s.d1(c, wl) + c
        c, bal2, g2 = s.moe2(c, p, e)
        noise = s.head(c)[:, 0]
        logv = s.vhead(c.detach())[:, 0].clamp(-12, 2) - 6.0
        return dict(xhat=yin - noise, logv=logv, z=z, stats=st, bal=bal1 + bal2, gates=g1, prompt=p)
