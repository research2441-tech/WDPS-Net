import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'legend.frameon': False, 'axes.grid': True, 'grid.color': '0.88', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'image.cmap': 'gray'})
GRAYS = ['0.0', '0.30', '0.50', '0.65', '0.78', '0.40', '0.20', '0.58']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']
HATCHES = ['', '////', '\\\\\\\\', 'xxx', '...', '---', '+++', 'ooo']
FIG = './figs'
