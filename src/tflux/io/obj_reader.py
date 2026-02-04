import pandas as pd
import numpy as np
from pathlib import Path


# Load .obj into DataFrame of vertex coordinate data.
def load_obj_vertices(file_path: Path) -> np.ndarray:
    '''

    Parameters
    ----------
    filename : Path
        Desired .obj file.

    Returns
    -------
    vertices_array_tyx : np.ndarray
        Vertices array with shape (total vertices, 3)

    '''
    vertices = []
    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(line.strip()[1:].split())
    
    df_xyz = pd.DataFrame(vertices, dtype='f4', columns=["x", "y", "z"][:len(vertices[0])])
    vertices_array_tyx = np.array(df_xyz.rename(columns={"x": "t", "z": "x"}))
    return vertices_array_tyx


