from fig_style import *
import numpy as np, json
import wdps_data as D
rows = [('AVIRIS: Jasper Ridge', 'jasper', 'S'), ('AVIRIS: San Diego', 'sandiego', 'S'), ('AVIRIS: Salton Sea', 'saltonsea', 'S'),
        ('AVIRIS: Salinas', 'salinas', 'X'), ('AVIRIS: Cuprite', 'cuprite', 'X'), ('AVIRIS: Indian Pines (220)', 'indianpines220', 'R'),
        ('APEX', 'apex', 'S'), ('SAMSON', 'samson', 'S'), ('Specim FENIX', 'fenix', 'S'),
        ('ROSIS: Pavia U./C.', 'paviaU', 'H'), ('HYDICE: Urban', 'urban', 'H'), ('HYDICE: DC Mall', 'dcmall', 'H'),
        ('Hyperion: Botswana', 'botswana', 'H'), ('CASI: MUUFL (64)', 'muufl64', 'H'), ('CASI: Houston', 'houston13', 'H'),
        ('AISA: Trento', 'trento', 'H'), ('CAVE', 'cave_peppers', 'H'), ('KAIST', 'kaist01', 'H'),
        ('CASI: MUUFL (72)', 'muufl72', 'R'), ('EnMAP: Potsdam', 'enmap', 'R')]
lab = {'S': 'always in training', 'X': 'cross-scene test', 'H': 'held-out sensor fold', 'R': 'real-noise test'}
fig, ax = plt.subplots(figsize=(7.0, 3.6))
for i, (name, key, role) in enumerate(rows[::-1]):
    wl = D.load(key)['wl']
    ls = {'S': '-', 'X': '-', 'H': '-', 'R': '-'}[role]
    col = {'S': '0.55', 'X': '0.35', 'H': '0.0', 'R': '0.0'}[role]
    ax.vlines(wl, i - 0.32, i + 0.32, color=col, lw=0.45)
    if role == 'R':
        noisy = D.load(key).get('noisy')
        if noisy is not None and len(noisy): ax.plot(wl[noisy], np.full(len(noisy), i + 0.45), 'v', ms=2.6, color='0.0', mec='none')
    ax.text(2580, i, f'{len(wl)}', va='center', fontsize=6.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.6)
for t, (_, _, role) in zip(ax.get_yticklabels(), rows[::-1]):
    t.set_fontweight('bold' if role == 'H' else 'normal'); t.set_fontstyle('italic' if role in 'XR' else 'normal')
ax.set_xlim(340, 2650); ax.set_ylim(-1.1, len(rows) - 0.2); ax.set_xlabel('Band-centre wavelength (nm)')
ax.text(2580, len(rows) - 0.35, 'B', fontsize=6.5, fontweight='bold')
ax.grid(axis='y', visible=False)
for x0, x1 in [(1340, 1450), (1800, 1960)]: ax.axvspan(x0, x1, color='0.93', zorder=0, lw=0)
ax.text(1395, -0.9, 'H$_2$O', ha='center', fontsize=6); ax.text(1880, -0.9, 'H$_2$O', ha='center', fontsize=6)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color='0.55', lw=1.2), Line2D([0], [0], color='0.35', lw=1.2), Line2D([0], [0], color='0.0', lw=1.2),
     Line2D([0], [0], marker='v', color='0.0', lw=0, ms=3.5)]
ax.legend(h, ['training-only sensors (grey)', 'cross-scene AVIRIS tests (italic)', 'held-out sensor folds (bold)', 'documented noisy bands (real-noise cubes)'],
          loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=6.3, handlelength=1.4, columnspacing=1.2)
plt.savefig(f'{FIG}/fig2_sensors.png')
