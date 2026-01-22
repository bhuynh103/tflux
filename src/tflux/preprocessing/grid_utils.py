# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:20:51 2025

@author: bhuyn
"""

import numpy as np
import tflux.pipeline.config as config
from tflux.dtypes import Junction, Grid
from scipy.ndimage import generic_filter


# Gridding
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