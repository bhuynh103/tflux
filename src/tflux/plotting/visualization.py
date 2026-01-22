# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:51:25 2025

@author: bhuyn
"""

import numpy as np
import matplotlib.pyplot as plt
import config
import analysis
from grid import Grid

### Visualization ###

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


def plot_vertices_3d(vertices, cmap=None, title=None, ax=None):
    """ Plot the vertices in 3D space (t, x, y). """
    t, y, x = vertices[:, 0], vertices[:, 1] * 1e6, vertices[:, 2] * 1e6

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)

    # Labels and title
    ax.set_xlabel("T (s)")
    ax.set_ylabel(u"X (μm)")
    ax.set_zlabel(u"Y (μm)")
    ax.set_title(f"{title}")
    ax.set_box_aspect(None, zoom=1)
    
    y_mid = (max(y) + min(y)) / 2
    x_range = max(x) - min(x)
    ax.set_zlim(y_mid - x_range/2, y_mid + x_range/2)
    
    if cmap is None:
        ax.scatter(t, x, y, c='gray', alpha=0.5)
    else:
        ax.scatter(t, x, y, c=y, cmap=cmap, s=5)
    
    return ax


def plot_xt_surface(grid: Grid, cmap=config.cmap1, ax=None):
    
    surface = grid.z
    x_range = grid.get_grid_range('x')
    t_range = grid.get_grid_range('t')
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    im = ax.imshow(surface.T, cmap=cmap, origin='lower', aspect='auto', extent=[0, x_range, 0, t_range])
    fig.colorbar(im, ax=ax, label='Amplitude (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('T (s)')
    
    return


def plot_amplitude_distribution(grid, bins=50, cmap=config.cmap1, ax=None):
    surface = grid.z
    amplitudes = surface.flatten()

    if ax is None:
        fig = plt.figure(figsize=(6, 4), dpi=300)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    n, bins, patches = ax.hist(
        amplitudes,
        bins=bins,
        edgecolor="black",
        linewidth=0.6,
    )
    
    bin_centers = 0.5 * (bins[:-1] - bins[1:])
    col = bins - min(bin_centers)
    col /= max(col)
    
    cm = plt.get_cmap(cmap)
    
    for c, p in zip(col, patches):
        p.set_facecolor(cm(c))
    
    ax.set_xlabel("Amplitude", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)

    return ax


def plot_3d_fft(mesh, log=False, log_residuals=False, include_best_fit=True, ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax = _ensure_ax_3d(ax, fig)
    
    if not log:
        x = mesh.x
        y = mesh.y
        z = mesh.z
    else:
        x = mesh.log_transform().x
        y = mesh.log_transform().y
        z = mesh.log_transform().z
        if log_residuals:
            z = mesh.log_transform().get_residuals()
            include_best_fit = False
            print("Ignoring best-fit plane for residual plot.")
    
    ax.plot_trisurf(
        x,
        y,
        z,
        cmap=config.cmap2, edgecolor='none', alpha=0.8
    )
        
    # Optionally plot the fitted plane
    if include_best_fit:
        # Create a coarse grid in (x, y) over the data range
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
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
    ax.set_xlabel("log q (1/m)")
    ax.set_ylabel("log omega (1/s)")
    ax.set_zlabel(r"log amp^2 ($m^4$)")
    ax.set_box_aspect(None, zoom=1)
    
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
    
    return ax1, ax2


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
        
    return ax1, ax2



### Analysis ###

def plot_2d_fft_slope(linreg, ax, scale=None):
        
    log_kx = linreg.x
    log_msd = linreg.y
    # log_std_err = linreg["yerr"]
    fit_tangent = linreg.x * linreg.m + linreg.int
    
    kx = 10 ** log_kx
    msd = 10 ** log_msd
    fit_tangent_10 = 10 ** fit_tangent    

    if ax == None:
        fig, ax = plt.subplots()
    
    # Plot the scatterplot
    
    # ax.errorbar(log_kx, log_msd, yerr=log_std_err, fmt="k.", c='red', capsize=0, elinewidth=0.5)
    ax.errorbar(kx, msd, yerr=0, fmt=".", c='black', capsize=0, elinewidth=0.5, ms=4, lw=0.25)
    # ax.plot(log_kx, fit_best, color='red')
    ax.plot(kx, fit_tangent_10, color='black')
    
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        match linreg.xlabel:
            case 'q':
                ax.set_xlabel("log q (1/m)")
            case 'w':
                ax.set_xlabel("log omega (1/s)")
        ax.set_ylabel("log amp squared (m^4)") # r"Log $\langle |u^2(q)| \rangle$ $(m^4)$"
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    return ax