# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:20:51 2025

@author: bhuyn
"""

import numpy as np
import pandas as pd
import config
from types import Junction

# Load .obj into DataFrame of vertex coordinate data.
def load_obj_vertices(filename):
    '''

    Parameters
    ----------
    filename : str
        Desired .obj file.

    Returns
    -------
    vertices_array_tyx : numpy.ndarray
        Vertices array with shape (total vertices, 3)

    '''
    vertices = []
    with open(filename, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(line.strip()[1:].split())
    
    df_xyz = pd.DataFrame(vertices, dtype='f4', columns=["x", "y", "z"][:len(vertices[0])])
    vertices_array_tyx = np.array(df_xyz.rename(columns={"x": "t", "z": "x"}))
    return vertices_array_tyx


def slice_vertices(vertices):
    '''
    Parameters
    ----------
    vertices : numpy.ndarray
        Vertices array with shape (total vertices, 3)

    Returns
    -------
    top_half : numpy.ndarray
        All vertices above the mean y.
    bottom_half : numpy.ndarray
        All vertices below the mean y.

    '''
    mean_y = vertices[:, 1].mean()
    # std_y = vertices[:, 1].std()
    top_half = vertices[vertices[:, 1] > mean_y]
    bottom_half = vertices[vertices[:, 1] <= mean_y]
    return top_half, bottom_half


def centralize_vertices(vertices):
    '''Centralize the vertices in the xy-plane'''
    if vertices[:, 1:].size > 1: 
        yx_center = vertices[:, 1:].mean(axis=0)  # Compute the mean of y and x
        centralized_vertices = vertices.copy()
        centralized_vertices[:, 1:] -= yx_center  # Shift y and x to center them
        return centralized_vertices
    else:
        raise ValueError("Invalid vertices.")


def find_best_orientation(vertices):
    '''
    Parameters
    ----------
    vertices : numpy.ndarray
        Vertices array with shape (total vertices, 3)
        Columns are in t, y, x order.

    Returns
    -------
    best_angle : float,
        Rotation angle around t-axis that minimizes the total y range.
    best_rotated_vertices : numpy.ndarray,
        Optimally rotated vertices array with shape (total vertices, 3)
        
    '''
    
    
    def rotate_vertices(vertices, angle_degrees):
        '''Rotates the vertices around the t-axis by the given angle.'''
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [1, 0, 0],  # t-axis remains unchanged
            [0, np.cos(angle_radians), -np.sin(angle_radians)],  # y and x rotated
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
        return vertices @ rotation_matrix.T

    centralized_vertices = centralize_vertices(vertices)

    best_angle = None
    min_y_range = float('inf')
    best_rotated_vertices = None

    # Iterate over angles from 0 to 360 degrees in 1-degree steps
    for angle in range(0, 360, 1):
        rotated = rotate_vertices(centralized_vertices, angle)
        y_range = rotated[:, 1].max() - rotated[:, 1].min()  # Compute the range of y

        if y_range < min_y_range:
            min_y_range = y_range
            best_angle = angle
            best_rotated_vertices = rotated

    return best_angle, best_rotated_vertices


# Transforms .obj into halves
def prepare_obj(file):
    '''

    '''
    vertices = load_obj_vertices(file)
        
    best_angle, best_vertices = find_best_orientation(vertices)
        
    best_vertices[:, 0] *= config.dt # pixels to seconds
    best_vertices[:, 1:] *= config.dx # pixels to meters

    top_half, bottom_half = slice_vertices(best_vertices)
    # top_half_centralized = centralize_vertices(top_half) # Introduces discontinuity
    # bottom_half_centralized = centralize_vertices(bottom_half)
    
    top_junc = Junction(vertices=top_half, is_top=True, filename=file)
    bot_junc = Junction(vertices=bottom_half, is_top=False, filename=file)
    
    return top_junc, bot_junc