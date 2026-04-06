import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tflux.pipeline import config
from tflux.plotting.axes import _ensure_ax_3d
from tflux.dtypes import Mesh
from tflux.plotting.rcparams import apply_3d_style
from tflux.plotting.plotting_utils import set_3d_axis_ticksize

def plot_3d_fft(mesh: Mesh, log=False, log_residuals=False, include_best_fit=True, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)
    
    if not log:
        x = mesh.q
        y = mesh.w
        z = mesh.z
    else:
        x = mesh.log_transform().q
        y = mesh.log_transform().w
        z = mesh.log_transform().z
        if log_residuals:
            z = mesh.log_transform().get_residuals()
            include_best_fit = False
            print("Ignoring best-fit plane for residual plot.")
    
    ax.plot_trisurf(
        x,
        y,
        z,
        cmap='Oranges', edgecolor='none', alpha=0.95
    )
        
    # Optionally plot the fitted plane
    if include_best_fit:
        # Create a coarse grid in (x, y) over the data range
        x_min, x_max = x.min() - 0.25, x.max()
        y_min, y_max = y.min() - 0.1, y.max()
        Xp, Yp = np.meshgrid(
            np.linspace(x_min, x_max, 30),
            np.linspace(y_min, y_max, 30),
        )
        Zp = mesh.a * Xp + mesh.b * Yp + mesh.c

        ax.plot_surface(
            Xp, Yp, Zp,
            alpha=0.3,
            edgecolor='none'
        )
    
    # Labels and title
    ax.tick_params(axis='x', pad=15)
    ax.tick_params(axis='y', pad=7.5)
    ax.tick_params(axis='z', pad=15)
    ax.set_xlabel(r"q ($m^{-1}$)", labelpad=15)
    ax.set_ylabel(r"$\omega$ ($s^{-1}$)", labelpad=15)
    ax.set_zlabel(r"$\langle | u^{2} | \rangle$ ($m^4$)", labelpad=30)
    ax.set_box_aspect(None, zoom=0.85)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    # tick_locations = [1, 2, 3] # Corresponding to 10^1, 10^2, 10^3 in log10 space
    # ax.set_zticks(tick_locations)
    formatter = ticker.FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(ticker.FixedLocator([4, 6]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-3, -1]))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[2, 4, 5], integer=True))
    
    apply_3d_style(ax)
    ax = set_3d_axis_ticksize(ax=ax)
    return ax


def plot_fft_vs_q_omega_loglog(fft, ax1=None, ax2=None):

    M, N = fft.shape
    q = np.fft.fftfreq(M, d=config.dx)  # Cycles per meter
    q = np.fft.fftshift(q)
    q_positive_mask = q > 0
    log_q = np.log10(q[q_positive_mask])
    
    omega = np.fft.fftfreq(N, d=config.dt)  # Cycles per meter
    omega = np.fft.fftshift(omega)
    omega_positive_mask = omega > 0
    log_omega = np.log10(omega[omega_positive_mask])
    
    fft_positive_q = fft[q_positive_mask]
    fft_positive = fft_positive_q[:, omega_positive_mask]
    fft_squared = np.abs(fft_positive) ** 2
    log_fft_squared = np.log10(fft_squared)
    
    if ax1 == None:
        fig1, ax1 = plt.subplots()
        
    if ax2 == None:
        fig2, ax2 = plt.subplots()
    
    valid_q_mask = log_q < config.TANGENT_CUTOFF
    valid_omega_mask = log_omega < config.TANGENT_CUTOFF_TIME
    
    max_m = np.sum(valid_q_mask)
    max_n = np.sum(valid_omega_mask)

    
    
    for m in range(max_m):
        t = m / (max_m - 1)      # 0 → 1
    
        if t < 0.5:
            # Green → Blue
            r = 0
            g = 1 - 2*t             # 1 → 0
            b = 2*t                 # 0 → 1
        else:
            # Blue → Purple
            r = 0.5 * (2*(t - 0.5))   # 0 → 0.5
            g = 0
            b = 1 - 0.5*(2*(t - 0.5)) # 1 → 0.5
    
        ax1.plot(log_omega,
                 log_fft_squared[m],
                 c=(r, g, b),
                 alpha=1,
                 linewidth=0.2)

    
    for n in range(max_n):
        t = n / (max_n - 1)      # 0 → 1
    
        if t < 0.5:
            # Orange → Red
            r = 1
            g = 0.5 * (1 - 2*t)     # 0.5 → 0
            b = 0
        else:
            # Red → Dark Red
            r = 1 - 0.5*(t - 0.5)*2  # 1 → 0.5
            g = 0
            b = 0
    
        ax2.plot(log_q,
                 log_fft_squared[:, n].flatten(),
                 c=(r, g, b),
                 alpha=1,
                 linewidth=0.2)
        
    ax1.set_xlabel('log omega (1/s)')
    ax1.set_ylabel('log amp squared (m^4)')
    ax1.set_facecolor('lightgray')
    ax1.grid()
    ax2.set_xlabel('log q (1/m)')
    ax2.set_ylabel('log amp squared (m^4)')
    ax2.set_facecolor('lightgray')    
    ax2.grid()
    
    return [ax1, ax2]


def plot_fft_vs_q_omega(fft, ax1=None, ax2=None, scale=None):

    M, N = fft.shape
    q = np.fft.fftfreq(M, d=config.dx)  # Cycles per meter
    q = np.fft.fftshift(q)
    q_positive_mask = q > 0
    log_q = np.log10(q[q_positive_mask])
    
    omega = np.fft.fftfreq(N, d=config.dt)  # Cycles per meter
    omega = np.fft.fftshift(omega)
    omega_positive_mask = omega > 0
    log_omega = np.log10(omega[omega_positive_mask])
    
    fft_positive_q = fft[q_positive_mask]
    fft_positive = fft_positive_q[:, omega_positive_mask]
    fft_squared = np.abs(fft_positive) ** 2
    
    if ax1 == None:
        fig1, ax1 = plt.subplots()
        
    if ax2 == None:
        fig2, ax2 = plt.subplots()
    
    valid_q_mask = log_q < config.TANGENT_CUTOFF
    valid_omega_mask = log_omega < config.TANGENT_CUTOFF_TIME
    
    max_m = np.sum(valid_q_mask)
    max_n = np.sum(valid_omega_mask)
    
    
    
    for m in range(max_m):
        t = m / (max_m - 1)      # 0 → 1
    
        if t < 0.5:
            # Green → Blue
            r = 0
            g = 1 - 2*t             # 1 → 0
            b = 2*t                 # 0 → 1
        else:
            # Blue → Purple
            r = 0.5 * (2*(t - 0.5))   # 0 → 0.5
            g = 0
            b = 1 - 0.5*(2*(t - 0.5)) # 1 → 0.5
    
        ax1.plot(omega[omega_positive_mask],
                 fft_squared[m],
                 c=(r, g, b),
                 alpha=1,
                 linewidth=0.4)

    
    for n in range(max_n):
        t = n / (max_n - 1)      # 0 → 1
    
        if t < 0.5:
            # Orange → Red
            r = 1
            g = 0.5 * (1 - 2*t)     # 0.5 → 0
            b = 0
        else:
            # Red → Dark Red
            r = 1 - 0.5*(t - 0.5)*2  # 1 → 0.5
            g = 0
            b = 0
    
        ax2.plot(q[q_positive_mask],
                 fft_squared[:, n].flatten(),
                 c=(r, g, b),
                 alpha=1,
                 linewidth=0.4)
        
    ax1.set_xlabel('omega (1/s)')
    ax1.set_ylabel('amp squared (m^4)')
    ax1.set_facecolor('lightgray')
    ax1.grid()
    # ax1.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0)) 
    if scale == 'log':
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
    
    ax2.set_xlabel('q (1/m)')
    ax2.set_ylabel('amp squared (m^4)')
    ax2.set_facecolor('lightgray')    
    ax2.grid()
    # ax2.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))
    if scale == 'log':
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(q[q_positive_mask].min()*0.8, q[q_positive_mask].max()*1.2)
        
    return [ax1, ax2]