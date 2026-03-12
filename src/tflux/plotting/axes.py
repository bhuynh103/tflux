import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _ensure_ax_3d(ax: plt.Axes | None, fig: plt.Figure, subplot_spec=None) -> Axes3D:
    """
    Ensure the given axes has a 3D projection, replacing it if necessary.
    Returns an Axes3D instance.
    """
    if ax is None:
        spec = subplot_spec if subplot_spec is not None else 111
        return fig.add_subplot(spec, projection='3d')

    if hasattr(ax, 'name') and ax.name == '3d':
        return ax

    pos = ax.get_subplotspec()
    fig.delaxes(ax)
    return fig.add_subplot(pos, projection='3d')