import math

import numpy as np
import pandas as pd
from libpyhat.data.spectra import Spectra
from libpyhat.data.spectrum import Spectrum


def band_minima(spectrum, low_endmember=None, high_endmember=None):
    """
    Given two end members, find the minimum observed value inclusively
    between them.

    Parameters
    ==========
    spectrum : nd.Array

    low_endmember : int
                    The low wavelength

    high_endmember : int
                    The high wavelength

    Returns
    =======
    minidx : int
             The wavelength of the minimum value

    minvalue : float
               The observed minimal value
    """

    if not low_endmember:
        low_endmember = 0
    if not high_endmember:
        high_endmember = spectrum.size
    else:
        high_endmember += 1

    sub_spectrum = spectrum[low_endmember:high_endmember]

    minvalue = np.amin(sub_spectrum)
    minidx = np.where(sub_spectrum == minvalue)[0]

    return minidx, minvalue


def band_center(spectrum, low_endmember=None, high_endmember=None, degree=3):

    if not low_endmember:
        low_endmember = 0
    if not high_endmember:
        high_endmember = spectrum.size
    else:
        high_endmember += 1

    sub_spectrum = spectrum[low_endmember:high_endmember]
    sub_spectrum_indices = list(range(len(sub_spectrum)))

    fit = np.polyfit(sub_spectrum_indices, sub_spectrum, degree)

    center_fit = np.polyval(fit, sub_spectrum_indices)
    center = band_minima(sub_spectrum)

    return center, center_fit


def band_area(spectrum, low_endmember=None, high_endmember=None):
    """
    Compute the area under the curve between two endpoints where the
    y-value <= 1.
    """
    if not low_endmember:
        low_endmember = 0
    if not high_endmember:
        high_endmember = spectrum.size
    else:
        high_endmember += 1

    sub_spectrum = spectrum[low_endmember:high_endmember]
    return np.trapz(np.where(sub_spectrum <= 1.0))


def band_asymmetry(spectrum, low_endmember=None, high_endmember=None, degree=3):
    """
    Compute the asymmetry of an absorption feature as
    (left_area - right_area) / total_area

    Parameters
    ----------
    spectrum : object

    low_endmember : int
        Bottom end of wavelengths to be obversed

    high_endmember : int
        Top end of wavelengths to be obversed

    Returns
    -------
    asymmetry : ndarray
        Array of percentage values of how asymmetrical the two halves of the spectrum are
        Where 100% is completely asymmetrical and 0 is completely symmetrical
    """

    if not low_endmember:
        low_endmember = 0
    if not high_endmember:
        high_endmember = spectrum.size
    else:
        high_endmember += 1

    sub_spectrum = spectrum[low_endmember:high_endmember]

    center, _ = band_center(sub_spectrum, degree=degree)
    center_idx = center[0][math.floor(len(center[0])/2)]

    if len(center[0]) % 2:
        area_left = band_area(sub_spectrum[:center_idx + 1])
    else:
        area_left = band_area(sub_spectrum[:center_idx])

    area_right = band_area(sub_spectrum[center_idx:])

    asymmetry = abs((area_left - area_right) / (area_left + area_right))
    return asymmetry


def analytics_series(spectrum, func, low_endmember=None, high_endmember=None):
    """
    Expects a spectrum object and calls func on it. Used in run_analytics to run
    functions across spectra


    Parameters
    ----------
    spectrum : Spectrum object

    func  : analytic functions

    low_endmember : int
        Bottom end of wavelengths to be obversed

    high_endmember : int
        Top end of wavelengths to be obversed

    Returns
    -------
    func : function return
    """

    if low_endmember:
        low_endmember = np.where(spectrum.data.index==low_endmember)[0]
    if high_endmember:
        high_endmember = np.where(spectrum.data.index==high_endmember)[0][0]

    ret = func(spectrum.data.values, low_endmember, high_endmember)
    if func is band_minima:
        wavelengths = spectrum.data.index[ret[0]]
        ret = wavelengths, ret[1]
    elif func is band_center:
        wavelengths = spectrum.data.index[ret[0][0]]
        center_fit = pd.Series(ret[1], index=spectrum.data.index)
        ret = (wavelengths, ret[0][1]), center_fit
    return ret


def run_analytics(obj, func, low_endmember=None, high_endmember=None):
    """
    Wrapper function to run any analytic func on a spectrum or spectra

    Parameters
    ----------
    obj : Spectrum or Spectra object

    func  : analytic functions

    low_endmember : int
        Bottom end of wavelengths to be obversed

    high_endmember : int
        Top end of wavelengths to be obversed

    Returns
    -------
    func : function return
    """

    if type(obj) is Spectrum:
        return analytics_series(obj, func, low_endmember, high_endmember)
    else:
        if func is band_center:
            index = obj.index[0:269]
            series = [analytics_series(obj[i], func, low_endmember, high_endmember) for i in obj.columns]
            centers, center_fits = zip(*series)
            return pd.Series(centers, index=obj.columns), pd.DataFrame(data = np.transpose(center_fits), columns = obj.columns, index = index)

        else:
            return pd.Series([analytics_series(obj[i], func, low_endmember, high_endmember) for i in obj.columns], index = obj.columns)
