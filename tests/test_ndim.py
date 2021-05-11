import conftest
import numpy as np

def test_ndim():
    np.testing.assert_array_almost_equal(conftest.n_dim(0), ([[[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]]]))