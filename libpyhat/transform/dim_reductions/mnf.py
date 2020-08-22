import numpy as np
import pandas as pd
from pysptools.noise import MNF
from libpyhat import Spectra

def mnf(data):
    '''
    Description: Minimum Noise Fraction (MNF) wrapper for pysptools implementation
    Rationale: Removes noise while preserving information
    '''
    # Convert Series to ndarray
    if issubclass(type(data), pd.Series) or isinstance(data, Spectra):
        np_data = data.to_numpy()
    elif issubclass(type(data), np.ndarray):
        np_data = data
    else:
        raise ValueError("Input for MNF must inherit from pd.Series or np.ndarray")

    # Ensure 3 dimensional input for MNF
    num_dimensions = len(np_data.shape)
    if num_dimensions == 2:
        # Add arbitrary 3rd dimension
        cube_data = np.expand_dims(np_data, axis=0)
    elif num_dimensions == 3:
        cube_data = np_data
    else:
        raise ValueError("Input must be 2 or 3 dimensional")

    # Open and apply MNF module
    pysp_mnf = MNF()
    res = pysp_mnf.apply(cube_data)

    # Return result in dimensionality of input
    if num_dimensions == 2:
        return np.squeeze(res, axis=0)
    else:
        return res
