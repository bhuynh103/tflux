# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:20:51 2025

@author: bhuyn
"""

import numpy as np
import tflux.pipeline.config as config
from tflux.utils.logging import get_logger
from tflux.dtypes import Junction, Grid
from scipy.ndimage import generic_filter

logger = get_logger(__name__)

# Gridding
def grid_xt(junc: Junction) -> Grid:
    vertices: np.ndarray = junc.vertices
    is_top: bool = junc.is_top
    
    if is_top:
        t, y, x = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    else:
        t, y, x = vertices[:, 0], vertices[:, 1] * -1, vertices[:, 2]
    
    x_range: float = max(vertices[:, 2]) - min(vertices[:, 2])
    t_range: float = max(vertices[:, 0]) - min(vertices[:, 0])
    
    # Dynamnically assign a grid size based on the cell length and sampling duration
    grid_size_x: int = int(x_range / config.dx) # x_range: 40 to 60 um, 200 to 300 pixel, dx = 0.205 um/pixel
    grid_size_t: int = int(t_range / config.dt)
    
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
    percent_zero: float = float(np.count_nonzero(count_grid == 0)) / (grid_size_x * grid_size_t)  
    # logger.info(f"Percent zero for junction = {percent_zero}")
    zero_padded_grid = np.nan_to_num(bin_grid)
    
    grid = Grid(x=x_bins, t=t_bins, z=zero_padded_grid, cts=count_grid, percent_zero=percent_zero)
    
    return grid


def interpolate_zeros(grid: Grid) -> Grid:
    z = grid.z
    cts = grid.cts
    
    # Define a filter to count non-zero neighbors
    def count_non_zero_neighbors(values: np.ndarray) -> int:
        non_zero_mask = values != 0
        return int(np.sum(non_zero_mask))
    
    # Apply the filter to count non-zero neighbors
    non_zero_neighbors: np.ndarray = generic_filter(
        z, 
        count_non_zero_neighbors, 
        size=config.WINDOW_SIZE, 
        mode='constant', 
        cval=0
    )
    majority_threshold: float = (config.WINDOW_SIZE ** 2) * config.MAJORITY_PERCENT
    
    # Identify sparse zeros: zero cells with many non-zero neighbors and insufficient counts
    sparse_zero_mask: np.ndarray = (non_zero_neighbors >= majority_threshold) & (cts < config.SUFFICIENT_COUNT)
    
    if config.WINDOW_SIZE % 2 == 0:
        print("Window size is even, adding 1 to make it odd for generic filter.")
        config.WINDOW_SIZE += 1 
    
    # Interpolate sparse zeros using averaging of non-zero neighbors
    def neighbor_mean(values: np.ndarray) -> float:
        if np.count_nonzero(values) <= 1:
            return 0.0
        non_zero_mask: np.ndarray = values != 0
        neighbor_average: float = float(np.mean(values[non_zero_mask]))
        return neighbor_average
    
    interpolated_grid: np.ndarray = z.copy()
    interpolated_grid[sparse_zero_mask] = generic_filter(
        z, 
        neighbor_mean, 
        size=config.WINDOW_SIZE, 
        mode='constant', 
        cval=0
    )[sparse_zero_mask]
    
    cts[sparse_zero_mask] += 1
    
    interpolated_grid_long: np.ndarray = interpolated_grid.astype(np.float64)
    
    grid.z = interpolated_grid_long
    grid.cts = cts
    
    return grid


def trim_grid(grid: Grid, crop_percent: float = config.CROP_PERCENT) -> Grid:
    x = grid.x
    z = grid.z

    # print(f"Pre trimming Length x: {len(grid.x)}")
    # print(f"Shape z: {grid.z.shape}")

    left = int(x.size * crop_percent / 2)
    right = int(x.size - left)
    grid.x = x[left:right]
    grid.z = z[left:right]
    
    return grid