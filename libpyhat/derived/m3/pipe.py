import numpy as np
import warnings

from . import pipe_funcs as pf

from .. import utils

@utils.warn_m3
def r750(data, **kwargs):
    """
    Name: R750
    Parameter: 0.75 um reflectance
    Formulation:
    R750 = R749
    Rationale: Reference I/F
    Bands: R749

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [749]
    return utils.generic_func(data, wavelengths, func = pf.reflectance_func, **kwargs)

@utils.warn_m3
def visnir(data, **kwargs):
    """
    Name: VISNIR
    Parameter: Visible-nearIR Ratio
    Formulation:
    VISUV = R699/R1579
    Rationale: Optical Maturity and mare-highland
    Bands: R699, R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [699, 1579]
    return utils.generic_func(data, wavelengths, func = pf.visnir_func, **kwargs)

@utils.warn_m3
def r950_750(data, **kwargs):
    """
    Name: R950_750
    Parameter: Ratio of 950nm to 750nm, mafic absorption
    Formulation:
    VISUV = R949/R749
    Rationale: Quick look at mafic absorption
    Bands: R749, R949

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [749, 950]
    return utils.generic_func(data, wavelengths, func = pf.r950_750_func, **kwargs)

@utils.warn_m3
def bd620(data, **kwargs):
    """
    Name: BD620
    Parameter: Band Depth at 620 nm
    Formulation:
    Numerator = R619
    Denominator = ((R749 - R419) / (749 - 419)) * (619 - 419) + R419
    BD620 = 1 - [Numerator/Denominator]
    Rationale: Possible Ti or Impact Melt
    Bands: R419, R619, R749

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [419, 619, 749]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

@utils.warn_m3
def bd950(data, **kwargs):
    """
    Name: BD950
    Parameter: Band Depth at 950 nm
    Formulation:
    Numerator = R949
    Denominator = ((R1579 - R749) / (1579 - 749)) * (949 - 749) + R749
    BD620 = 1 - [Numerator/Denominator]
    Rationale: OPX Comparison with Kaguya
    Bands: R749, R949, R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [749, 949, 1579]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

@utils.warn_m3
def bd1050(data, **kwargs):
    """
    Name: BD1050
    Parameter: Band Depth at 1050 nm
    Formulation:
    Numerator = R1049
    Denominator = ((R1579 - R749) / (1579 - 749)) * (1049 - 749) + R749
    BD620 = 1 - [Numerator/Denominator]
    Rationale: OLV Comparison with Kaguya
    Bands: R749, R1049, R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [749, 1049, 1579]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

@utils.warn_m3
def bd1250(data, **kwargs):

    """
    Name: BD1250
    Parameter: Band Depth at 1250 nm
    Formulation:
    Numerator = R1249
    Denominator = ((R1579 - R749) / (1579 - 749)) * (1249 - 749) + R749
    BD620 = 1 - [Numerator/Denominator]
    Rationale: PLAG Comparison with Kaguya
    Bands: R749, R1249, R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [749, 1249, 1579]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

@utils.warn_m3
def bd2800(data, **kwargs):
    """
    Name: BD2800
    Parameter: Band Depth at 2800 nm
    Formulation:
    Numerator = R2817
    Denominator = ((R2976 - R2697) / (2976 - 2697)) * (2817 - 2897) + R2697
    BD620 = 1 - [Numerator/Denominator]
    Rationale: OH absorption at 2.8um (Color ramp: dark is weak, white is strong)
    Bands: R2697, R2817, R2976

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [2697, 2817, 2976]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

@utils.warn_m3
def r1580(data, **kwargs):
    """
    Name: R1580
    Parameter: 1.6 um reflectance
    Formulation:
    R1580 = R1579
    Rationale: IR Albedo
    Bands: R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1579]
    return utils.generic_func(data, wavelengths, func = pf.reflectance_func, **kwargs)





@utils.warn_m3
def bdi1000(data, **kwargs):
    """
    Name: BDI1000
    Parameter: 1 um integrated band depth
    Formulation:
    BDI1000 = Sum with n values 0-26: (1 - [R(789 + 20n) / Rc(789 + 20n)])
    Rationale: Fe Mineralogy
    Bands: R789 - R1308 (in steps of 20)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    return utils.bdi_generic(data, 27, 789, 20)

@utils.warn_m3
def twoum_ratio(data, **kwargs):
    """
    Name: 2um_Ratio
    Parameter: 2 um ratio
    Formulation:
    2um_Ratio = R1578/R2538
    Rationale: N/A
    Bands: R1578, R2538

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1578, 2538]
    return utils.generic_func(data, wavelengths, func = pf.twoum_ratio_func, **kwargs)

@utils.warn_m3
def bdi2000(data, **kwargs):
    """
    Name: BDI2000
    Parameter: 2 um integrated band depth
    Formulation:
    BDI1000 = Sum with n values 0-21: (1 - [R(1658 + 40n) / Rc2(1658 + 40n)])
    Rationale: Fe Mineralogy
    Bands: R1658 - R2498 (in steps of 40)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    return utils.bdi_generic(data, 22, 1658, 40)

@utils.warn_m3
def thermal_ratio(data, **kwargs):
    """
    Name: Thermal_Ratio
    Parameter: N/A
    Formulation:
    Thermal_Ratio = R2538/2978
    Rationale: N/A
    Bands: R2538, R2978

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [2538, 2978]
    return utils.generic_func(data, wavelengths, func = pf.thermal_ratio_func, **kwargs)

@utils.warn_m3
def bd3000(data, **kwargs):
    """
    Name: BD3000
    Parameter: 3 um band depth using 2um continuum
    Formulation:
    Numerator = R2978
    Denominator = ((R2538 - R1578) / (2538 - 1578)) * (2978 - 1578) + R1578
    BD620 = 1 - [Numerator/Denominator]
    Rationale: H2O
    Bands: R1578, R2538, R2978

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1578, 2538, 2978]
    return utils.generic_func(data, wavelengths, func = pf.bd3000_func, **kwargs)

def r540(data, **kwargs):
    """
    Name: R540
    Parameter: 0.55 um reflectance
    Formulation:
    R750 = R539
    Rationale: Reference I/F
    Bands: R539

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [539]
    return utils.generic_func(data, wavelengths, func = pf.reflectance_func, **kwargs)

def visslope(data, **kwargs):
    """
    Name: Vis_Slope
    Parameter: UV-visible continuum slope
    Formulation:
    Vis_Slope = (R749 - R419) / (749 - 419)
    Rationale: UV-Vis Slope (%/nm)
    Bands: R419, R749

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [419, 749]
    return utils.generic_func(data, wavelengths, func = pf.visslope_func, **kwargs)

def oneum_slope(data, **kwargs):
    """
    Name: 1um_Slope
    Parameter: continuum slope between 0.70 and 1.6 um
    Formulation:
    1um_Slope = (R1579 - R699) / (1579 - 699)
    Rationale: Vis-NIR Slope (%/nm)
    Bands: R699, R1579

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [699, 1579]
    return utils.generic_func(data, wavelengths, func = pf.oneum_slope_func, **kwargs)

def r2780(data, **kwargs):
    """
    Name: R2780
    Parameter: 2.8 um reflectance
    Formulation:
    R750 = R2778
    Rationale: Reference I/F
    Bands: R2778

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [2778]
    return utils.generic_func(data, wavelengths, func = pf.reflectance_func, **kwargs)

def olindex(data, **kwargs):
    """
    Name: OLINDEX
    Parameter: Olivine Index
    Formulation:
    slope = (R1750 - R650) / (1750 - 650)
    a = 0.1 * [(slope * (860-650) + R650) / R860]
    b = 0.5 * [(slope * (1047-650) + R650) / R1047]
    c = 0.25 * [(slope * (1230-650) + R650) / R1230]
    OLINDEX = a + b + c
    Rationale: Olivine will be strongly positive
    Bands: R650, R860, R1047, R1230, R1750

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [650, 860, 1047, 1230, 1750]
    return utils.generic_func(data, wavelengths, func = pf.olindex_func, **kwargs)

def bd1900(data, **kwargs):
    """
    Name: BD1900
    Parameter: Band Depth at 1900 nm: low Ca pyroxene index
    Formulation:
    Numerator = R1898
    Denominator = ((R2498 - R1408) / (2498 - 1408)) * (1898 - 1408) + R1408
    BD620 = 1 - [Numerator/Denominator]
    Rationale: pyroxene will be positive; favors LCP
    Bands: R1408, R1898, R2498

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1408, 1898, 2498]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

def bd2300(data, **kwargs):
    """
    Name: BD2300
    Parameter: Band Depth at 2300 nm: low Ca pyroxene index
    Formulation:
    Numerator = R2298
    Denominator = ((R2578 - R1578) / (2578 - 1578)) * (2298 - 1578) + R1578
    BD620 = 1 - [Numerator/Denominator]
    Rationale: pyroxene will be positive; favors LCP
    Bands: R1578, R2298, R2578

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1578, 2298, 2578]
    return utils.generic_func(data, wavelengths, func = pf.bd_func, pass_wvs=True, **kwargs)

def twoum_slope(data, **kwargs):
    """
    Name: 2um_Slope
    Parameter: continuum slope between 1.6 and 2.5 um
    Formulation:
    21um_Slope = (R2538 - R1578) / (2538 - 1578)
    Rationale: NIR Slope
    Bands: R1578, R2538

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """
    wavelengths = [1578, 2538]
    return utils.generic_func(data, wavelengths, func = pf.twoum_slope_func, **kwargs)
