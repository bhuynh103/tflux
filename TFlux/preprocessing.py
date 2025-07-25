# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:20:51 2025

@author: bhuyn
"""

import numpy as np
import pandas as pd

# File Handling
def load_obj(filename):
    """
    Reads an OBJ file and returns its elements as a NumPy array.
    """
    vertices = []
    with open(filename, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(line.strip()[1:].split())
    
    df = pd.DataFrame(vertices, dtype='f4', columns=["x", "y", "z"][:len(vertices[0])])
    return np.array(df.rename(columns={"x": "t", "z": "x"}))

# Vertex Slicing/Rotating
def slice_vertices(vertices):
    """
    Splits vertices into top and bottom halves based on mean y value.
    """
    mean_y = vertices[:, 1].mean()
    top_half = vertices[vertices[:, 1] > mean_y]
    bottom_half = vertices[vertices[:, 1] <= mean_y]
    return top_half, bottom_half

def find_best_orientation(vertices):
    """
    Finds the best orientation to minimize the range of y values by rotating the object
    around the t-axis in 15-degree increments.

    Parameters:
    - vertices: np.ndarray
        A NumPy array of vertices with columns [t, y, x].

    Returns:
    - best_angle: float
        The angle in degrees that minimizes the range of y values.
    - rotated_vertices: np.ndarray
        The vertices rotated to the best orientation.
    """
    def rotate_vertices(vertices, angle_degrees):
        """Rotates the vertices around the t-axis by the given angle."""
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [1, 0, 0],  # t-axis remains unchanged
            [0, np.cos(angle_radians), -np.sin(angle_radians)],  # y and x rotated
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
        return vertices @ rotation_matrix.T

    # Initialize variables to track the best angle and minimum range
    best_angle = None
    min_y_range = float('inf')
    best_rotated_vertices = None

    # Iterate over angles from 0 to 360 degrees in 15-degree steps
    for angle in range(0, 360, 15):
        rotated = rotate_vertices(vertices, angle)
        y_range = rotated[:, 1].max() - rotated[:, 1].min()  # Compute the range of y

        if y_range < min_y_range:
            min_y_range = y_range
            best_angle = angle
            best_rotated_vertices = rotated

    return best_angle, best_rotated_vertices

def centralize_and_rotate_obj(vertices, angle_degrees):
    """
    Centralizes the OBJ vertices in the xy-plane and rotates them around the t-axis.

    Parameters:
    - vertices: np.ndarray
        A NumPy array of vertices with columns [t, y, x].
    - angle_degrees: float
        The angle in degrees by which to rotate the vertices around the t-axis.

    Returns:
    - rotated_vertices: np.ndarray
        A NumPy array of the centralized and rotated vertices.
    """
    # Centralize the vertices in the xy-plane
    xy_center = vertices[:, 1:].mean(axis=0)  # Compute the mean of y and x
    centralized_vertices = vertices.copy()
    centralized_vertices[:, 1:] -= xy_center  # Shift y and x to center them

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix around the t-axis
    rotation_matrix = np.array([
        [1, 0, 0],  # t-axis remains unchanged
        [0, np.cos(angle_radians), -np.sin(angle_radians)],  # y and x rotated
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Apply rotation to centralized vertices
    rotated_vertices = centralized_vertices @ rotation_matrix.T
    return rotated_vertices