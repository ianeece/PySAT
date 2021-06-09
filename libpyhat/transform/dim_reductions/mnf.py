import numpy as np
import pandas as pd
from pysptools.noise import Whiten
from libpyhat import Spectra
from sklearn.decomposition import PCA

#adapted from pysptools to take n_components
def mnf(M,n_components):
    w = Whiten()
    wdata = w.apply(M)
    h, w, numBands = wdata.shape
    X = np.reshape(wdata, (w * h, numBands))
    pca = PCA(n_components=n_components)
    mnf = pca.fit_transform(X)
    mnf = np.reshape(mnf, (h, w, n_components))
    return mnf

class MNF():
    def __init__(self, n_components = 4):
        self.n_components = n_components

    def fit_transform(self,data):
        '''
        Description: Minimum Noise Fraction (MNF) wrapper for pysptools implementation
        Rationale: Removes noise while preserving information
        '''
        # Convert Series to ndarray
        if issubclass(type(data), pd.Series) or isinstance(data, Spectra) or issubclass(type(data), pd.DataFrame):
            np_data = data.to_numpy()
        elif issubclass(type(data), np.ndarray):
            np_data = data
        else:
            raise ValueError("Input for MNF must inherit from pd.Series or np.ndarray")
    
        cube_data, num_dimensions = make_3d(np_data)

        #check data shape
        #if cube_data.shape[0]*cube_data.shape[1]>cube_data.shape[2]:

        mnf_result = mnf(cube_data, self.n_components)

        # Return result in dimensionality of input
        if num_dimensions == 2:
            return np.squeeze(mnf_result)
        else:
            return mnf_result
        # else:
        #     raise ValueError("For MNF, # of samples must be greater than # of spectral channels")
        #
def make_3d(data):
    # Ensure 3 dimensional input for MNF
    num_dimensions = len(data.shape)
    if num_dimensions == 2:
        # Add arbitrary 3rd dimension
        cube_data = np.expand_dims(data, axis=0)
    elif num_dimensions == 3:
        cube_data = data
    else:
        raise ValueError("Input must be 2 or 3 dimensional")
    return cube_data, num_dimensions


