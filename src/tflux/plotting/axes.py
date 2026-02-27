import matplotlib.pyplot as plt

def _ensure_ax_3d(ax, fig, subplot_spec=None):
    '''
    Change projection of 2D ax to 3D

    '''
    if ax == None:
        if subplot_spec == None:
            fig.add_subplot(111, projection='3d')
        else:
            fig.add_subplot(subplot_spec, projection='3d')
        return ax
    
    if hasattr(ax, 'name') and ax.name == '3d':
        return ax
    
    pos = ax.get_subplotspec()
    fig.delaxes(ax)
    ax = fig.add_subplot(pos, projection='3d')
    return ax