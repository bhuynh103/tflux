# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:44:22 2025

@author: bhuyn
"""

import pandas as pd
import numpy as np
import os
import config
import preprocessing
from scipy.ndimage import generic_filter
from scipy.fft import fft2, fftshift
from scipy.stats import linregress
from junction import Junction, Sample
from mesh import Mesh
from grid import Grid
from linreg import LinReg
            


# Gridding Data
def grid_xt(junc: Junction):
    '''
    
    '''
    vertices = junc.vertices
    is_top = junc.is_top
    
    if is_top:
        t, y, x = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    else:
        t, y, x = vertices[:, 0], vertices[:, 1] * -1, vertices[:, 2]
    
    t_range = max(vertices[:, 0]) - min(vertices[:, 0])
    x_range = max(vertices[:, 2]) - min(vertices[:, 2])
    
    # Dynamnically assign a grid size based on the cell length and sampling duration
    grid_size_x = int(x_range / config.dx) # x_range: 40 to 60 um, 200 to 300 pixel, dx = 0.205 um/pixel
    grid_size_t = int(t_range / config.dt)
    
    # Construct bins and grids
    x_bins = np.linspace(min(x), max(x), grid_size_x)
    t_bins = np.linspace(min(t), max(t), grid_size_t)
    bin_grid = np.full((grid_size_x, grid_size_t), fill_value=np.nan)
    count_grid = np.zeros_like(bin_grid, dtype=int)
    
    # Assign each vertex to an x-index and a t-index
    x_indices = np.digitize(x, x_bins, right=True)
    t_indices = np.digitize(t, t_bins, right=True)

    # Iterate over vertices by index, populate grids
    for xi, ti, yi in zip(x_indices, t_indices, y):
        if 0 <= xi < bin_grid.shape[0] and 0 <= ti < bin_grid.shape[1]:
            count_grid[xi, ti] += 1
            if np.isnan(bin_grid[xi, ti]):
                bin_grid[xi, ti] = yi
            else:
                bin_grid[xi, ti] += yi

    # Take the mean of each grid based on the counts
    np.divide(bin_grid, count_grid, out=bin_grid, where=~np.isnan(bin_grid))
    
    # Track the number of zeroes in a grid.
    percent_zero = (np.count_nonzero(count_grid == 0)) * 100 / (grid_size_x * grid_size_t)  
    
    print(f"    Percent Zeros: { (np.count_nonzero(count_grid == 0)) * 100/ (grid_size_x * grid_size_t):.2f}%")
    # print(f"    Percent One: { (np.count_nonzero(count_grid == 1)) * 100/ (grid_size_x * grid_size_t):.2f}%")
    # print(f"    Percent Two: { (np.count_nonzero(count_grid == 2)) * 100/ (grid_size_x * grid_size_t):.2f}%")
    # print(f"    Percent Three+: { (np.count_nonzero(count_grid >= 3)) * 100/ (grid_size_x * grid_size_t):.2f}%")
    
    zero_padded_grid = np.nan_to_num(bin_grid)
    
    grid = Grid(x=x_bins, y=t_bins, z=zero_padded_grid, cts=count_grid, grid_type='default', percent_zero=percent_zero)
    junc.grid = grid
    
    return junc


def interpolate_zeros(grid: Grid):
    '''

    '''
    
    z = grid.z
    cts = grid.cts
    
    # Define a filter to count non-zero neighbors
    def count_non_zero_neighbors(values):
        non_zero_mask = values != 0
        non_zero_neighbors_count = np.sum(non_zero_mask)
        return non_zero_neighbors_count

    # Apply the filter to count non-zero neighbors
    non_zero_neighbors = generic_filter(z, count_non_zero_neighbors, size=config.WINDOW_SIZE, mode='constant', cval=0)
    majority_threshold = (config.WINDOW_SIZE ** 2) * config.MAJORITY_PERCENT
    
    # Identify sparse zeros: zero cells with many non-zero neighbors and sufficient counts
    sparse_zero_mask = (non_zero_neighbors >= majority_threshold) & (cts < config.SUFFICIENT_COUNT)

    if config.WINDOW_SIZE % 2 == 0:
        print("Window size is even, adding 1 to make it odd for generic filter.")
        config.WINDOW_SIZE += 1 
        
    # Interpolate sparse zeros using averaging of non-zero neighbors
    def neighbor_mean(values):
        if np.count_nonzero(values) <= 1:
            return 0
        non_zero_mask = values != 0
        neighbor_average = np.mean(values[non_zero_mask])
        return neighbor_average

    interpolated_grid = z.copy()
    interpolated_grid[sparse_zero_mask] = generic_filter(z, neighbor_mean, size=config.WINDOW_SIZE, mode='constant', cval=0)[sparse_zero_mask]
    cts[sparse_zero_mask] += 1
    interpolated_grid_long = interpolated_grid.astype(np.float64) 
    
    grid.z = interpolated_grid_long
    grid.cts = cts
    
    # print(f"Interpolated grid with a sufficient count threshold of {sufficient_count}.")
    return grid


def trim_grid(grid):
    x = grid.x
    z = grid.z
    
    index_trim = config.CROP_PERCENT * 0.01
    
    left = int(x.size * index_trim)
    right = int(x.size - left)
    grid.z = z[left:right]
    
    return grid

# FFT


def fft_to_mesh(grid: Grid):
    if grid.grid_type == 'fourier':        
        z2_mesh = grid.z_tilde
        z2_mesh_transpose = z2_mesh.T
        q_mesh, w_mesh = np.meshgrid(grid.q, grid.w, indexing='ij')
        mesh = Mesh(q_mesh, w_mesh, z2_mesh_transpose, False)
        return mesh


def linreg_on_fft(grid: Grid):
    if grid.grid_type == 'fourier':
        linreg_q = grid.grid_to_linreg_over('q')
        linreg_w = grid.grid_to_linreg_over('w')
        return linreg_q, linreg_w


def tension_interpolation(interp):
    return (config.boltzmann_constant * config.room_temp) / ((10 ** (interp + 4.5)))

# Stats

def calc_sample_metrics(sample, metrics: list[str]):
    N = len(sample.juncs)
    if N >= 1:
        for metric in metrics:
            mean, std = sample.find_average(metric)
            print(f'{metric} = {mean} +/- {std}')
    

### Batch Processing via Directory ###

def process_surface(junc: Junction):
    # results = {}
    
    junc = grid_xt(junc)  # Construct the Grid object
    junc.grid = interpolate_zeros(junc.grid)
    # junc.grid = trim_grid(junc.grid)
    junc.grid = junc.grid.fourier_transform(shift_fft=True, square_fft=True)
    print(junc.grid.shifted, junc.grid.squared)
    junc.linreg_q, junc.linreg_w = linreg_on_fft(junc.grid)
    
    junc.mesh = fft_to_mesh(junc.grid)  # Contruct the Mesh object

    junc.mesh = junc.mesh.apply_masks(denoise=True)  # Slice above positive frequency and below noise floor
    junc.mesh = junc.mesh.find_loglog_gradient()
    
    
    # results = {
    #     "is_top": junc.is_top,
    #     "vertices": junc.vertices,
    #     "count_grid": junc.grid.cts,
    #     "fft": junc.grid.z_tilde,
    #     "fft_shifted": junc.grid.z_tilde, # Need to confirm
    #     "linreg_q": junc.linreg_q,
    #     "linreg_w": junc.linreg_w,
    #     "mesh": junc.mesh,
    #     "grid": junc.grid,
    # }
    return junc


def process_files(directory_path):
    # results = {}
    sample = Sample()
    obj_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.obj')]

    if not obj_files:
        print(f"No OBJ files found in directory: {directory_path}")
        return

    for file in obj_files:
        file_basename = os.path.basename(file)
        print(f"\nProcessing file: {file_basename}")
        
        top_junc, bot_junc = preprocessing.prepare_obj(file)  # Pixels to seconds and meters
        # print(f"  Cell Length: {x_range * 10 ** 6:2f} microns")
        
        # results[f"top:{file_basename}"] = process_surface(top_junc)
        # results[f"bot:{file_basename}"] = process_surface(bot_junc)
        
        top_junc_processed = process_surface(top_junc)
        bot_junc_processed = process_surface(bot_junc)
        
        sample = sample.append_junction(top_junc_processed)
        sample = sample.append_junction(bot_junc_processed)
    
    # N = len(results)
    
    # results_df = pd.DataFrame(columns=["linreg_q_slope", "linreg_q_int", "mesh_a", "mesh_b"])
    
    # linreg_list = [junc.linreg for junc in results.values()]
    # slope_list = [junc.linreg.m for junc in results.values()]
    # intercept_list = [junc.linreg.b for junc in results.values()]
    # # tension_list = [data["tension"] for data in results.values()]
    # a_list = [junc.mesh.a for junc in results.values()]
    # b_list = [junc.mesh.b for junc in results.values()]
    
    return sample


def get_directories_in_path(path):
    """
    Returns a list of directory names within the specified path.
    """
    directories = []
    try:
        # Get all entries (files and directories) in the path
        entries = os.listdir(path)
        for entry in entries:
            full_path = os.path.join(path, entry)
            # Check if the entry is a directory
            if os.path.isdir(full_path):
                directories.append(full_path)
    except FileNotFoundError:
        print(f"Error: The path '{path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to access '{path}'.")
    return directories


# def pair_process_directories(directory):  
#     treatment_directory_list = get_directories_in_path(directory)
#     scatterplot_dfs = []
#     alpha_dfs = []
    
#     for treatment_directory in treatment_directory_list:
#         group_list = get_directories_in_path(treatment_directory)
        
#         for group_index, group_dir_path in enumerate(group_list):
#             sample = process_files(group_dir_path)
#             slope_array_positive = np.array(slope_list) * -1
            
#             for junction_index, linreg in enumerate(linreg_list):
#                 x = linreg["log_x"]
#                 y = linreg["log_y"]
#                 sample = np.array([f"{group_dir_path}" for i in range(0, len(x))])
#                 treatment = np.array([f"{os.path.basename(treatment_directory)}" for i in range(0, len(x))])
#                 junction = np.array([junction_index for i in range(0, len(x))])
#                 group_index_list = np.array([group_index for i in range(0, len(x))])
#                 group_labels = np.where(group_index_list == 0, "control", "experimental")
#                 scatterplot_df_chunk = pd.DataFrame(data=list(zip(x, y, treatment, group_labels, junction, sample)), columns=["log_x", "log_y", "treatment", "group", "junction", "sample"])
#                 scatterplot_dfs.append(scatterplot_df_chunk)
            
#             sample_alpha = np.array([f"{group_dir_path}" for i in range(0, len(slope_list))])
#             treatment_alpha = np.array([f"{os.path.basename(treatment_directory)}" for i in range(0, len(slope_list))])
#             group_index_list_alpha = np.array([group_index for i in range(0, len(slope_list))])
#             group_label_alpha = np.where(group_index_list_alpha == 0, "control", "experimental")
#             alpha_df_chunk = pd.DataFrame(data=list(zip(slope_array_positive, treatment_alpha, group_label_alpha, sample_alpha)), columns=["alpha", "treatment", "group", "sample"])
#             alpha_dfs.append(alpha_df_chunk)
    
#     scatterplot_df = pd.concat(scatterplot_dfs, ignore_index=True)
#     scatterplot_df.to_csv("paired_figure_data.csv", index=False)
    
#     alpha_df = pd.concat(alpha_dfs, ignore_index=True)
#     alpha_df.to_csv("paired_alpha_data.csv", index=False)

#     return scatterplot_df, alpha_df
