import numpy as np
from scipy.ndimage import gaussian_filter

def mpsnr(x, y):
    mse = ((x - y) ** 2).reshape(-1, x.shape[2]).mean(0)
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))

def mssim(x, y, s=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2; out = []
    for b in range(x.shape[2]):
        a, c = x[:, :, b].astype(np.float64), y[:, :, b].astype(np.float64)
        ma, mc = gaussian_filter(a, s), gaussian_filter(c, s)
        va = gaussian_filter(a * a, s) - ma ** 2; vc = gaussian_filter(c * c, s) - mc ** 2; cov = gaussian_filter(a * c, s) - ma * mc
        out.append((((2 * ma * mc + C1) * (2 * cov + C2)) / ((ma ** 2 + mc ** 2 + C1) * (va + vc + C2))).mean())
    return float(np.mean(out))

def sam(x, y):
    X, Y = x.reshape(-1, x.shape[2]).astype(np.float64), y.reshape(-1, y.shape[2]).astype(np.float64)
    c = (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())

def ergas(x, y):
    X, Y = x.reshape(-1, x.shape[2]), y.reshape(-1, y.shape[2])
    rmse = np.sqrt(((X - Y) ** 2).mean(0)); mu = np.maximum(X.mean(0), 1e-3)
    return float(100 * np.sqrt(np.mean((rmse / mu) ** 2)))

def all_metrics(clean, est):
    est = np.clip(est, 0, 1)
    return dict(psnr=mpsnr(clean, est), ssim=mssim(clean, est), sam=sam(clean, est), ergas=ergas(clean, est))
