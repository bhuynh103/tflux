import pandas as pd
import numpy as np
from pathlib import Path


def convert_cartesian_element_to_array(element_data: list[str], relabel=True) -> np.ndarray:

    # For our data, the x and z coordinates are swapped in the .obj file, so we relabel them here to be consistent.
    relabel_mapping = {
        'obj.x': 't', 
        'obj.y': 'y',
        'obj.z': 'x'
    }

    df_xyz = pd.DataFrame(element_data, dtype='f4', columns=["obj.x", "obj.y", "obj.z"][:len(element_data[0])])
    if relabel:
        df_xyz = df_xyz.rename(columns=relabel_mapping)
    
    vertices_array = np.array(df_xyz)
    return vertices_array


# Load .obj into DataFrame of vertex coordinate data.
def load_obj(file_path: Path, element: str, relabel: bool=True) -> np.ndarray:
    '''

    Parameters
    ----------
    file_path : Path
        Desired .obj file.
    element : str
        Element name to be used in the returned array. vertices = 'v ', normals = 'vn ', faces = 'f '
    relabel : bool, optional
        Whether to relabel the x and z coordinates to t and x respectively. The default is True.

    Returns
    -------
    element_data : np.ndarray
        Array of vertex coordinates, normals, or faces depending on the element parameter.

    '''
    element_data = []

    element_mapping = {
        'vertices': 'v ',  
        'normals': 'vn ',
        'faces': 'f ',
    }

    with open(file_path, 'r') as obj_file:

        element_code = element_mapping[element]
        element_code_len = len(element_mapping[element])

        for line in obj_file:
            if line.startswith(element_code):
                element_data.append(line.strip()[element_code_len:].split())

    match element:
        case 'vertices':
            return convert_cartesian_element_to_array(element_data, relabel=relabel)
        case 'normals':
            return convert_cartesian_element_to_array(element_data, relabel=relabel)
        case 'faces':
            return element_data