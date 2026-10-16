"""Fig. 1: WDPS-Net architecture (grayscale). Solid arrows = forward computation; dashed arrows = reverse-mode gradients.
Equation numbers are read from paper/eqmap.json (written by the document builder) so the figure matches the text."""
import json, os, re
from fig_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams['axes.grid'] = False
EQ = json.load(open('./paper/eqmap.json')) if os.path.exists('./paper/eqmap.json') else {}
def eq(*keys):
    n = [EQ.get(k, '?') for k in keys]
    return f'({n[0]})' if len(n) == 1 else f'({n[0]})–({n[1]})'

fig = plt.figure(figsize=(7.16, 4.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 58.5); ax.axis('off')
B = {}
def box(k, x, y, w, h, title, body='', fc='1.0', tsize=6.5, bsize=5.7, ls=1.22):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.7', fc=fc, ec='0.0', lw=0.8, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=tsize, fontweight='bold', zorder=3)
        ax.text(x + w / 2, y + h - 3.0, body, ha='center', va='top', fontsize=bsize, zorder=3, linespacing=ls)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=tsize, fontweight='bold', zorder=3, linespacing=1.2)
    B[k] = (x, y, w, h)
def side(k, s, f=0.5):
    x, y, w, h = B[k]; p = 0.2
    return {'l': (x - p, y + f * h), 'r': (x + w + p, y + f * h), 't': (x + f * w, y + h + p), 'b': (x + f * w, y - p)}[s]
def seg(pts, ls='-', lw=0.8, col='0.0', head=True):
    for a, b in zip(pts[:-1], pts[1:-1] + ([] if head else [pts[-1]])):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, ls=ls, zorder=4, solid_capstyle='butt', dash_capstyle='butt')
    if head:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=6.5, lw=lw, color=col, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=4))
def lab(x, y, s, size=5.8, ha='center', va='center', col='0.0'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=col, zorder=6, bbox=dict(fc='white', ec='none', pad=0.25))
def dot(x, y): ax.add_patch(Circle((x, y), 0.45, color='0.0', zorder=5))

# inputs
box('lam', 1.0, 48.5, 11.0, 7.5, 'Band centres\n$\\boldsymbol{\\lambda}=(\\lambda_1,\\ldots,\\lambda_B)$', fc='0.90', tsize=6.1)
box('Y', 1.0, 29.0, 11.0, 10.0, 'Observed cube\n$\\mathbf{Y}\\in\\mathbb{R}^{H\\times W\\times B}$\nany $B$, any $\\boldsymbol{\\lambda}$', fc='0.90', tsize=6.1)
# encoders
box('wenc', 16.0, 48.5, 23.0, 8.0, 'Wavelength encoder', f'Fourier code + MLP $\\rightarrow\\mathbf{{e}}_b$  {eq("wcode")}\ngap $|\\lambda_b-\\lambda_{{b-1}}|/\\delta_0$ for the scan step {eq("step")}')
box('denc', 16.0, 33.0, 23.0, 11.0, 'Degradation encoder', f'$\\hat\\sigma_b,\\hat\\varsigma_b,\\hat\\delta_b,\\hat\\iota_b,\\hat\\rho_b$ + high-pass CNN\n$\\rightarrow$ band prompts $\\mathbf{{p}}_b$  {eq("mad","prompt")}\ntrained end to end, no labels')
box('sub', 16.0, 21.5, 23.0, 8.0, 'Unmixing-subspace prior', f'3$\\times$3 median $\\rightarrow$ eigenvectors $\\mathbf{{U}}_r$\n$\\mathcal{{P}}_4(\\mathbf{{Y}}),\\ \\mathcal{{P}}_{{10}}(\\mathbf{{Y}})$  {eq("proj")}')
# trunk
ty, th = 3.0, 10.5
box('stem', 1.0, ty, 10.5, th, 'Stem', '3$\\times$3$\\times$3 conv on\n[$\\mathbf{Y}$, mask, $\\mathcal{P}_4$, $\\mathcal{P}_{10}$]\n$\\rightarrow C$ channels/band', bsize=5.4)
box('film', 14.5, ty, 9.5, th, 'FiLM', f'$\\mathbf{{F}}_b\\odot(1+\\boldsymbol{{\\gamma}}_b)$\n$+\\boldsymbol{{\\beta}}_b$  {eq("film")}')
box('moe1', 27.0, ty, 12.5, th, 'Routed experts', f'top-2 of 4 per band\nspatial, column,\nspectral, dilated  {eq("gate","moe")}', bsize=5.4)
box('ss', 43.0, ty, 24.0, th, 'Wavelength-stepped encoder-decoder', f'5 selective spectral recurrences {eq("wsr","step")}\nalternating direction; full and 1/2 resolution\nspatial bi-scan at the bottleneck {eq("ssm")}, {eq("merge")}\ninternal full-resolution skip', fc='0.94', bsize=5.3)
box('moe2', 70.5, ty, 10.0, th, 'Routed experts', f'top-2 of 4\n{eq("gate","moe")}')
box('head', 84.5, ty, 14.0, th, 'Head', f'residual $\\hat{{\\mathbf{{R}}}}$\nlog-variance $\\hat{{\\mathbf{{S}}}}$ (detached)\n$\\hat{{\\mathbf{{X}}}}=\\mathbf{{Y}}-\\hat{{\\mathbf{{R}}}}$  {eq("out")}', bsize=5.4, ls=1.5)
# outputs, reference, objectives
box('out', 84.5, 21.5, 14.0, 8.5, 'Outputs', 'restored $\\hat{\\mathbf{X}}$\nband reliability $\\bar u_b$', ls=1.5)
box('ref', 84.5, 38.0, 14.0, 8.0, 'Clean $\\mathbf{X}$', 'source sensors only', fc='0.90')
box('loss', 56.0, 32.0, 23.5, 14.5, 'Objectives', f'source sensors (clean pairs):\n$\\mathcal{{L}}_{{\\rm rec}}+\\alpha_1\\mathcal{{L}}_{{\\rm nll}}+\\alpha_2\\mathcal{{L}}_{{\\rm bal}}$  {eq("total")}\ntarget sensor (noisy cubes only):\nre-degraded pseudo-clean + $\\mathcal{{L}}_{{\\rm um}}$  {eq("ssa")}', fc='0.97', bsize=5.5)

# ---------------- forward computation
seg([side('lam', 'r'), side('wenc', 'l', 0.58)])
seg([side('Y', 'r', 0.78), side('denc', 'l', 0.45)])
seg([side('Y', 'r', 0.22), (14.0, side('Y', 'r', 0.22)[1]), (14.0, side('sub', 'l', 0.55)[1]), side('sub', 'l', 0.55)])
seg([side('Y', 'b', 0.35), side('stem', 't', 0.36)])
seg([side('sub', 'b', 0.12), (18.76, 16.0), (8.6, 16.0), side('stem', 't', 0.72)])
lab(13.6, 16.0, '$\\mathcal{P}_r(\\mathbf{Y})$', size=5.6)
# conditioning bus: e_b and gaps from the wavelength encoder, p_b from the degradation encoder
bus_y, xe, xp = 18.0, 41.5, 43.0
seg([side('wenc', 'r', 0.45), (xe, side('wenc', 'r', 0.45)[1]), (xe, bus_y)], head=False)
seg([side('denc', 'r', 0.25), (xp, side('denc', 'r', 0.25)[1]), (xp, bus_y)], head=False)
ax.plot([21.5, 77.5], [bus_y, bus_y], color='0.0', lw=0.8, zorder=4)
dot(xe, bus_y); dot(xp, bus_y)
lab(xe + 0.7, 45.6, '$\\mathbf{e}_b$, gaps', size=5.6, ha='left')
lab(xp + 0.9, 30.0, '$\\mathbf{p}_b$', size=5.8, ha='left')
for x, k in [(21.5, 'film'), (35.0, 'moe1'), (48.0, 'ss'), (77.5, 'moe2')]:
    seg([(x, bus_y), (x, ty + th + 0.2)])
lab(60.0, bus_y + 1.25, 'conditioning: $\\mathbf{e}_b$, $\\mathbf{p}_b$, $\\Delta_b$ gaps', size=5.5)
seg([side('stem', 'r'), side('film', 'l')]); seg([side('film', 'r'), side('moe1', 'l')])
seg([side('moe1', 'r'), side('ss', 'l')])
seg([side('ss', 'r'), side('moe2', 'l')])
seg([side('moe2', 'r'), side('head', 'l')])
seg([side('head', 't', 0.6), side('out', 'b', 0.6)])
seg([side('out', 'l', 0.5), (67.75, side('out', 'l', 0.5)[1]), side('loss', 'b', 0.5)])
seg([side('ref', 'l', 0.5), (side('loss', 'r')[0], side('ref', 'l', 0.5)[1])])
# ---------------- reverse-mode gradients (dashed)
G = dict(ls=(0, (3.2, 1.8)), lw=0.9, col='0.30')
seg([side('loss', 'b', 0.2), (60.7, ty + th + 0.2)], **G)
seg([side('loss', 'l', 0.12), (side('denc', 'r')[0], side('loss', 'l', 0.12)[1])], **G)
lab(52.4, 26.0, 'reverse-mode gradient\n$\\nabla_{\\theta}\\mathcal{L}$ into all\ntrainable modules', size=5.4, col='0.2')
# legend
ax.plot([55.5, 59.5], [56.0, 56.0], color='0.0', lw=0.8); ax.text(60.2, 56.0, 'forward computation', fontsize=5.8, va='center')
ax.plot([77.0, 81.0], [56.0, 56.0], color='0.30', lw=0.9, ls=(0, (3.2, 1.8))); ax.text(81.7, 56.0, 'reverse-mode gradient', fontsize=5.8, va='center')
plt.savefig(f'{FIG}/fig1_architecture.png', dpi=400)
