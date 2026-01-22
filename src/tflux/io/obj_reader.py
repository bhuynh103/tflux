import pandas as pd

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