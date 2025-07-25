# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:51:25 2025

@author: bhuyn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift
from scipy.stats import linregress
import config

### Visualization ###

def plot_xt_surface(surface, t_range, x_range, index):
    plt.figure(figsize=(8, 6))
    plt.imshow(surface.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, x_range, 0, t_range])
    plt.colorbar(label='Amplitude (m)')
    plt.title(f"Gridded Top-Half of Cell {index}")
    plt.xlabel("X (m)")
    plt.ylabel("T (s)")
    plt.show()


def plot_vertices_3d(vertices, index):
    """ Plot the vertices in 3D space (t, x, y). """
    t, y, x = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(t, x, y, c=y, cmap='viridis', s=5)

    # Labels and title
    ax.set_title(f"Cell {index} XYT Image Mesh")
    ax.set_xlabel("T (s)")
    ax.set_ylabel("X (m)")
    ax.set_zlabel("Y (m)")
    ax.set_box_aspect(None, zoom=0.85)
    
    plt.show()


def plot_3d_fft_loglog(ft_shifted, t_range, x_range, index):
    magnitude = np.abs(ft_shifted)
    grid_size_x, grid_size_t = magnitude.shape
    
    # Find frequencies using fftfreq(n=num_points, d=sample_spacing)
    freqs_x = fftshift(np.fft.fftfreq(grid_size_x, d=x_range / grid_size_x))
    freqs_t = fftshift(np.fft.fftfreq(grid_size_t, d=t_range / grid_size_t))
    fx, ft = np.meshgrid(freqs_x, freqs_t)

    log_magnitude = np.log10(magnitude + 1e-10)
    positive_mask = (fx > 0) & (ft > 0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        np.log10(fx[positive_mask].flatten()),
        np.log10(ft[positive_mask].flatten()),
        log_magnitude[positive_mask].flatten(),
        cmap='coolwarm', edgecolor='none', alpha=0.8
    )

    # Labels and title
    ax.set_title(f"Log-Log Plot of 2D Fourier Transform of XYT Cell {index} Image Mesh")
    ax.set_xlabel("Spatial Frequency (1/m)", labelpad=10)
    ax.set_ylabel("Temporal Frequency (1/s)", labelpad=5)
    ax.set_zlabel("Amplitude (m² * s)")
    ax.set_box_aspect(None, zoom=0.85)

    # Get current ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    z_ticks = ax.get_zticks()

    # Filter ticks to keep only integers
    x_ticks = [tick for tick in x_ticks if tick.is_integer()]
    y_ticks = [tick for tick in y_ticks if tick.is_integer()]
    z_ticks = [tick for tick in z_ticks if tick.is_integer()]
    
    # Convert ticks to exponentiated labels
    x_tick_labels = [f"$10^{{{int(tick)}}}$" for tick in x_ticks]
    y_tick_labels = [f"$10^{{{int(tick)}}}$" for tick in y_ticks]
    z_tick_labels = [f"$10^{{{int(tick)}}}$" for tick in z_ticks]

    # Set new ticks with exponentiated labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    
    plt.show()


### Analysis ###

def plot_2d_fft_slope(linreg, index):

    log_kx = linreg["x"]
    log_msd = linreg["y"]
    log_std_err = linreg["yerr"]
    
    fit_best = linreg["fit_best"]
    fit_tangent = linreg["fit_tangent"]
    
    # Plot the scatterplot
    plt.figure(figsize=(8, 6))
    plt.errorbar(log_kx, log_msd, yerr=log_std_err, fmt="k.", capsize=0, elinewidth=0.5)
    plt.plot(log_kx, fit_best, color='red')
    plt.plot(log_kx, fit_tangent, color='orange')
    plt.title(f"Log Amplitude Squared vs. Log Wavenumber Averaged over Frequency (Cell {index})")
    plt.xlabel("Log Wavenumber")
    plt.ylabel("Log Amplitude Squared")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    plt.show()