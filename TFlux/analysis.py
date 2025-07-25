# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:44:22 2025

@author: bhuyn
"""

import numpy as np
import os
import config
import preprocessing
from scipy.ndimage import generic_filter
from scipy.fft import fft2, fftshift
from scipy.stats import linregress


# Gridding Data
def grid_xt(vertices, x_range, is_top=True):
    """
    Grids x-t data and assigns maximum or minimum y-values as amplitude.
    """
    t, y, x = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    grid_size = int(x_range / config.dx) # x_range: 40 to 60 um, 200 to 300 pixel, dx = 0.205 um/pixel
    
    x_bins = np.linspace(min(x), max(x), grid_size)
    t_bins = np.linspace(min(t), max(t), grid_size)
    grid = np.full((grid_size - 1, grid_size - 1), np.nan)
    count_grid = np.zeros_like(grid, dtype=int)

    x_indices = np.digitize(x, x_bins) - 1
    t_indices = np.digitize(t, t_bins) - 1

    for xi, ti, yi in zip(x_indices, t_indices, y):
        if 0 <= xi < grid.shape[0] and 0 <= ti < grid.shape[1]:
            count_grid[xi, ti] += 1
            if np.isnan(grid[xi, ti]):
                grid[xi, ti] = yi
            else:
                grid[xi, ti] = max(grid[xi, ti], yi) if is_top else min(grid[xi, ti], yi)
                
    percent_zero = (np.count_nonzero(count_grid == 0)) * 100 / (grid_size**2)
    
    print(f"    Percent Zeros: { (np.count_nonzero(count_grid == 0)) * 100/ (grid_size**2):.2f}%")
    print(f"    Percent One: { (np.count_nonzero(count_grid == 1)) * 100/ (grid_size**2):.2f}%")
    print(f"    Percent Two: { (np.count_nonzero(count_grid == 2)) * 100/ (grid_size**2):.2f}%")
    print(f"    Percent Three+: { (np.count_nonzero(count_grid >= 3)) * 100/ (grid_size**2):.2f}%")

    return np.nan_to_num(grid), count_grid, percent_zero, grid_size

# Interpolation
def interpolate_sparse_zeros(grid, count_grid, majority_percent, sufficient_count):
    """
    Interpolates sparse zeros in a grid based on neighbor density and point sufficiency.
    """
    # Define a filter to count non-zero neighbors
    def count_non_zero_neighbors(values):
        center = values[len(values) // 2]
        return np.sum(values != 0) if center == 0 else -1

    # Apply the filter to count non-zero neighbors
    neighbor_counts = generic_filter(grid, count_non_zero_neighbors, size=config.WINDOW_SIZE, mode='constant', cval=0)
    majority = (config.WINDOW_SIZE ** 2) * config.MAJORITY_PERCENT
    
    # Identify sparse zeros: zero cells with many non-zero neighbors and sufficient counts
    sparse_zero_mask = (neighbor_counts >= majority) & (count_grid < sufficient_count)

    # Interpolate sparse zeros using weighted averaging of neighbors
    def weighted_average(values):
        center = values[len(values) // 2]
        if center == 0:
            weights = (values != 0).astype(float)
            return np.sum(values * weights) / np.sum(weights) if np.sum(weights) > 0 else 0
        return center

    interpolated_grid = grid.copy()
    interpolated_grid[sparse_zero_mask] = generic_filter(
        grid, weighted_average, size=config.WINDOW_SIZE, mode='constant', cval=0
    )[sparse_zero_mask]
    
    print(f"Interpolated grid with a sufficient count threshold of {sufficient_count}.")
    return interpolated_grid

# Fourier Transform
def compute_2d_fft(surface):
    ft = fft2(surface)
    return ft, fftshift(ft)


def linreg_on_fft(fft_shifted, x_range, grid_size):
    
    # Compute wavenumbers
    kx = np.fft.fftfreq(grid_size - 1, d=x_range / grid_size)
    kx = np.fft.fftshift(kx)  # Shift zero frequency to the center
    
    amplitude_squared = np.abs(fft_shifted) ** 2
    # Average the amplitude over temporal frequencies ###
    msd = amplitude_squared.mean(axis=0)  # Mean amplitude squared over temporal frequencies
    
    # Find standard errors
    msd_std = amplitude_squared.std(axis=0)
    msd_std_err = msd_std / (amplitude_squared.shape[0] ** 0.5) # Number of omegas
    
    # Filter to include only positive wavenumbers
    positive_mask = kx > 0
    kx_positive = kx[positive_mask]
    msd_positive = msd[positive_mask]
    msd_std_err_positive = msd_std_err[positive_mask]

    # Log-transform data
    log_kx = np.log10(kx_positive)
    log_msd = np.log10(msd_positive)
    log_std_err = msd_std_err_positive / (msd_positive * np.log(10) + 1e-10) # Log Transform on Uncertainty

    # Fit a linear regression to the log-log data
    slope, intercept, _, _, _ = linregress(log_kx, log_msd) # r, p_value, std_err
    slope_neg, intercept_neg, _, _, _ = linregress(log_kx[log_kx < config.TANGENT_CUTOFF], log_msd[log_kx < config.TANGENT_CUTOFF])
    
    fit_best = slope * log_kx + intercept
    fit_neg = slope_neg * log_kx + intercept_neg
    
    linreg = {"x": log_kx,
              "y": log_msd,
              "yerr": log_std_err,
              "fit_best": fit_best,
              "fit_tangent": fit_neg,
              }
    
    return slope_neg, intercept_neg, linreg


def tension_interpolation(intercept):
    return (config.boltzmann_constant * config.room_temp) / (10 ** (intercept))


# Stats

def calculate_metrics(slope_list, intercept_list, tension_list):
    print(slope_list)
    N = len(slope_list)
    metrics_array = np.array([slope_list, intercept_list, tension_list])
    metrics_mean = np.mean(metrics_array, axis=1, keepdims=True)
    metrics_std = np.std(metrics_array, axis=1, ddof=1, mean=metrics_mean)
    metrics_std_err = metrics_std / np.sqrt(N)
    return metrics_mean.flatten(), metrics_std_err
    

### Batch Processing via Directory ###

def process_files_in_directory(directory, dx, dt):
    results = {}
    obj_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.obj')]

    if not obj_files:
        print(f"No OBJ files found in directory: {directory}")
        return results

    for file in obj_files:
        file_name = os.path.basename(file)
        print(f"\nProcessing file: {file_name}")
        vertices = preprocessing.load_obj(file)
        
        best_angle, best_rotated_vertices = preprocessing.find_best_orientation(vertices)
        best_vertices = preprocessing.centralize_and_rotate_obj(vertices, best_angle)
        
        best_vertices[:, 0] *= dt # pixels to seconds
        best_vertices[:, 1:] *= dx # pixels to meters

        t_range = max(best_vertices[:, 0]) - min(best_vertices[:, 0])
        x_range = max(best_vertices[:, 2]) - min(best_vertices[:, 2])
        
        top_half, _ = preprocessing.slice_vertices(best_vertices)
        count_vertices = len(top_half)
        print(f"  Cell Length: {x_range * 10 ** 6:2f} microns")
        xt_surface, count_grid, percent_zero, grid_size = grid_xt(top_half, x_range, is_top=True)
        print(f"  Grid Size: {grid_size}")
        
        xt_surface_interpolated = interpolate_sparse_zeros(xt_surface, count_grid, config.MAJORITY_PERCENT, config.SUFFICIENT_COUNT)

        ft, ft_shifted = compute_2d_fft(xt_surface_interpolated)
        
        slope, intercept, linreg = linreg_on_fft(ft_shifted, x_range, grid_size)
        
        tension = tension_interpolation(intercept)

        results[file_name] = {
            "vertices": best_vertices,
            "t_range": t_range,
            "x_range": x_range,
            "grid_size": grid_size,
            "xt_surface": xt_surface_interpolated,
            "count_grid": count_grid,
            "fft": ft,
            "fft_shifted": ft_shifted,
            "percent_zero": percent_zero,
            "linreg": linreg,
            "slope": slope,
            "intercept": intercept,
            "tension": tension,
        }
        
    N = len(results)
    
    return results, grid_size, N