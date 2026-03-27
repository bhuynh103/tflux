import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

BASE_PARAMS = {
    'font.family': 'Arial',
    'font.size': 14,
    'axes.labelsize': 32,
    'axes.titlesize': 18,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.constrained_layout.use': True
}

PARAMS_3D = {
    'axes.labelsize': 28,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'ztick.labelsize': 20,
}

mpl.rcParams.update(BASE_PARAMS)


def apply_3d_style(ax: Axes3D) -> None:
    ax.xaxis.label.set_size(PARAMS_3D['axes.labelsize'])
    ax.yaxis.label.set_size(PARAMS_3D['axes.labelsize'])
    ax.zaxis.label.set_size(PARAMS_3D['axes.labelsize'])
    ax.tick_params(axis='both', labelsize=PARAMS_3D['xtick.labelsize'])