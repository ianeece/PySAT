import inspect
from functools import wraps

import numpy as np
import warnings
from .m3 import pipe_funcs as pf

def generic_func(data, wavelengths, kernels={}, func=None, axis=0, pass_wvs=False, **kwargs):
    """
    Using some form of data and a wavelength array. Get the bands associated
    wtih each wavelength in wavelengths, create a subset of bands based off
    of those wavelengths then hand the subset to the function.

    Parameters
    ----------
    data : ndarray
           (x, y, z) 3 dimensional numpy array of a spectra image

    wv_array : iterable
               A list of all possible wavelengths for a given spectral image

    wavelengths : iterable
                  List of wavelengths to use for the function

    Returns
    ----------
    : func
      Returns the result from the given function
    """
    if kernels:
        subset = []
        wvs = data.wavelengths

        for k, v in kernels.items():
            s = sorted(np.abs(np.array(wvs)-k).argsort()[:v])
            subset.append(np.median(data.iloc[s, :, :], axis=axis))
        if len(subset) == 0:
            subset = subset[0]
    else:
        subset = data.loc[wavelengths, :, :]

    for i in subset:
        i[i == data.no_data_value] = 0

    if pass_wvs:
        return func(subset, wavelengths, **kwargs)

    return func(subset, **kwargs)

def compute_b_a(wavelengths):
    '''
    Given a set of three wavelengths compute their b and a values as per
    the Viviano Beck CRISM Derived Products paper
    (Revised CRISM spectral parameters and summary
    products based on the currently detected
    mineral diversity on Mars)

    Parameters
    ----------
    wavelengths : iterable
        A list of three wavelength values

    Returns
    -------
    b : float
        b value from the paper

    a : float
        a value from the paper
    '''
    wavelengths.sort()
    lambda_s, lambda_c, lambda_l = wavelengths

    b = (lambda_c - lambda_s) / (lambda_l - lambda_s)
    a = 1.0 - b
    return b, a

def compute_slope(x1, x2, y1, y2):
    '''
    Computes slope given two points on a line

    Parameters
    ----------
    x1 : float
        First points x value

    x2 : float
        Second points x value

    y1 : float
        First points y value

    y2 : float
        Second points y value

    Returns
    -------
    : float
        Slope between the two points
    '''

    return (y2 - y1) / (x2 - x1)

def line_fit(slope, x, b):
    '''
    Finds the y value for a given x using a given slope and y intercept

    Parameters
    ----------
    slope : float
        Slope of a line

    x : float
        Point along the x axis of a line

    b : float
        Y intercept of a line

    Returns
    -------
    : float
        Y coordinate corresponding to the given x
    '''
    return (slope * x) + b

def calc_bdi_band(data, iteration, initial_band, step, **kwargs):
    """
    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    wv_array : ndarray
               (n,1) array of wavelengths that correspond to the p
               dimension of the data array

    iteration : int
                Number of steps to add to the new band calculation

    initial_band : int
                   Initial band to use to calculate the new band

    step : int
           Length between bands to calculate

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    y = initial_band + (step * iteration)
    wv_array = data.wavelengths
    vals = np.abs(data.wavelengths - y)
    minidx = np.argmin(vals)
    wavelengths = [wv_array[minidx - 3], y, wv_array[minidx + 3]]
    wvlims = [wavelengths[0], y, wavelengths[-1]]
    return generic_func(data, wavelengths, func=pf.bdi_func, pass_wvs=wvlims, **kwargs)

def bdi_generic(data, upper_limit, initial_band, step):
    """
    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    wv_array : ndarray
               (n,1) array of wavelengths that correspond to the p
               dimension of the data array

    upper_limit : int
                  Upper limit on the number of wavelengths to be used

    initial_band : int
                   The band to use as a starting point to extract the other
                   0 to upper_limit bands
    step : int
           The step size inbetween the 0 to upper limit bands

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    limit = range(0, upper_limit)
    band_list = [1 - calc_bdi_band(data, i, initial_band, step) for i in limit]

    return np.sum(band_list, axis = 0)

def warn_m3(m3_func, *args, **kwargs):
    @wraps(m3_func)
    def call_warn(*args, **kwargs):
        warnings.warn('Parameters involving some of the visible wavelengths ( < 600 nm) are not recommended for use. Parameters modeled after Clementine data are also not recommended. Original parameter estimates for OH and H2O should NOT be included.')
        return m3_func(*args, **kwargs)
    return call_warn

def get_derived_funcs(package):
    derived_funcs = {}

    modules = inspect.getmembers(package, inspect.ismodule)

    for module in modules:
        if "funcs" not in module[0]:
            derived_funcs = dict(inspect.getmembers(module[1], inspect.isfunction), **derived_funcs)

    return derived_funcs
