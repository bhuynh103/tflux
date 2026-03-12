import numpy as np
from tflux.dtypes import Junction

# Vertices
def slice_vertices(vertices) -> tuple[np.ndarray, np.ndarray]:
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


def centralize(vertices: np.ndarray) -> np.ndarray:
    '''Centralize the vertices in the xy-plane'''
    if vertices[:, 1:].size > 1: 
        yx_center = vertices[:, 1:].mean(axis=0)  # Compute the mean of y and x
        centralized_vertices = vertices.copy()
        centralized_vertices[:, 1:] -= yx_center  # Shift y and x to center them
        return centralized_vertices
    else:
        raise ValueError("Invalid vertices.")


def find_best_orientation(vertices: np.ndarray) -> np.ndarray:  #(vertices: np.ndarray , normals: np.ndarray)
    '''
    Parameters
    ----------
    vertices : numpy.ndarray
        Vertices array with shape (total vertices, 3)
        Columns are in t, y, x order.
    normals : numpy.ndarray
        Normals array with shape (total vertices, 3)

    Returns
    -------
    best_rotated_vertices : numpy.ndarray,
        Optimally rotated vertices array with shape (total vertices, 3)
        
    '''
    
    def rotate(cartesian_array, angle_degrees) -> np.ndarray:
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [1, 0, 0],  # t-axis remains unchanged
            [0, np.cos(angle_radians), -np.sin(angle_radians)],  # y and x rotated
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
        return cartesian_array @ rotation_matrix.T

    centralized_vertices = centralize(vertices)

    min_y_range = float('inf')
    best_rotated_vertices = None

    # Iterate over angles from 0 to 360 degrees in 1-degree steps
    for angle in range(0, 360, 1):
        rotated_vertices = rotate(centralized_vertices, angle)
        y_range = rotated_vertices[:, 1].max() - rotated_vertices[:, 1].min()  # Compute the range of y

        if y_range < min_y_range:
            min_y_range = y_range
            best_angle = angle
            best_rotated_vertices = rotated_vertices

    # best_y_sum = 0
    # best_angle = None

    # for angle in range(0, 360, 1):
    #     rotated_normals = rotate(normals, angle)
    #     y_sum = np.abs(rotated_normals[:, 1]).sum()  # Compute the sum of y components of the normals

    #     if y_sum > best_y_sum:
    #         best_y_sum = y_sum
    #         best_angle = angle
    #         best_rotated_vertices = rotate(centralized_vertices, best_angle)
        
    return best_rotated_vertices


def reorient_junction(junc: Junction) -> Junction:
    junc.vertices = centralize(junc.vertices)
    junc.vertices = find_best_orientation(junc.vertices)
    return junc