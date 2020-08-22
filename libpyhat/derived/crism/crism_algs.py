import numpy as np

from . import crism_funcs as cf
from ..utils import generic_func


def r770(data, **kwargs):
    """
    Name: R770
    Parameter: 0.77micron reflectance
    Formulation: R770
    Rationale: Higher value more dusty or icy
    Caveats: Sensitive to slope effects, clouds

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [770]
    kernels = {770: 5}

    return generic_func(data, wv, kernels=kernels, func = cf.rockdust1_func, pass_wvs = False, **kwargs)


def rbr(data, **kwargs):
    """
    Name: RBR
    Parameter: Red/Blue Ratio
    Formulation: R770 / R440
    Rationale: Higher value indicates more npFeOx
    Caveats: Sensitive to dust in atmosphere

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [440, 770]
    kernels = {440:5,
               770:5}

    return generic_func(data, wv, kernels=kernels, func=cf.rockdust2_func, **kwargs)


def bd530(data, use_kernels = True, **kwargs):
    """
    NAME: BD530
    PARAMETER: 0.53 micron band depth
    FORMULATION: 1 - (R530/(a*R709+b*R440))
    RATIONALE: Crystalline ferric minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [440, 530, 716]
    kernels = {}

    if use_kernels:
        wv = [440, 530, 614]
        kernels[440] = 5
        kernels[530] = 5
        kernels[614] = 5

    return generic_func(data, wv, func = cf.bd_func1, kernels = kernels, pass_wvs = True, **kwargs)


def sh600(data, use_kernels = True, **kwargs):
    """
    NAME: SH600
    PARAMETER: 0.60 micron shoulder height
    FORMULATION: 1 - (a * R530 + b * R709) / R600
    FORMULATION (with kernels): 1 - (a * R533 + b * R716) / R600
    RATIONALE: select ferric minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [530, 600, 709]
    kernels = {}

    if use_kernels:
        wv = [533, 600, 716]
        kernels[533] = 5
        kernels[600] = 5
        kernels[716] = 3

    return generic_func(data, wv, func = cf.sh_func, pass_wvs = True, kernels = kernels, **kwargs)

def sh770(data, **kwargs):
    """
    NAME: SH770
    PARAMETER: 0.77 micron shoulder height
    FORMULATION (with kernels): 1 - (a * R716 + b * R860) / R775
    RATIONALE: select ferric minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [716, 775, 860]
    kernels = {716: 3,
               775: 5,
               860: 5}

    return generic_func(data, wv, func = cf.sh_func, pass_wvs = True, kernels = kernels, **kwargs)


def bd640(data, use_kernels = True, **kwargs):
    """
    NAME: BD640
    PARAMETER: 0.64 micron band depth
    FORMULATION: 1 - (R648 / (a * R600 + b * R709))
    FORMULATION (with kernels): 1 - (R624 / (a * R600 + b * R760))
    RATIONALE: select ferric minerals, especially maghemite

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [600, 648, 709]
    kernels = {}

    if use_kernels:
        wv = [600, 624, 760]
        kernels[600] = 5
        kernels[624] = 3
        kernels[760] = 5

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def bd860(data, use_kernels = True, **kwargs):
    """
    NAME: BD860
    PARAMETER: 0.86 micron band depth
    FORMULATION: 1 - (R860 / (a * R800 + b * R984))
    FORMULATION (with kernels): 1 - (R860 / (a * R755 + b * R977))
    RATIONALE: select ferric minerals ('hematite band')

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [800, 860, 984]
    kernels = {}

    if use_kernels:
        wv = [755, 860, 977]
        kernels[755] = 5
        kernels[860] = 5
        kernels[977] = 5

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def bd920(data, use_kernels = True, **kwargs):
    """
    NAME: BD920
    PARAMETER: 0.92 micron band depth
    FORMULATION: 1 - ( R920 / (a * R800 + b * R984) )
    FORMULATION (with kernels): 1 - ( R920 / (a * R807 + b * R984) )
    RATIONALE: select ferric minerals ('Pseudo BDI1000 VIS')

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [800,920,984]
    kernels = {}

    if use_kernels:
        wv = [807, 920, 984]
        kernels[807] = 5
        kernels[920] = 5
        kernels[984] = 5

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def rpeak1(data, **kwargs):
    """
    NAME: RPEAK1
    PARAMETER: Reflectance peak 1
    FORMULATION: Wavelength where first derivative = 0 of fifth-order \
            polynomial fit to reflectances at all valid VNIR wavelengths
    RATIONALE: crystalline Fe+2 or Fe+3 minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wvs = data.wavelengths
    vnir_mask = (wvs > 400) * (wvs < 1000)
    vnir_wvs = wvs[vnir_mask]

    return generic_func(data, vnir_wvs, func = cf.rpeak1_func, pass_wvs = True, **kwargs)

def bdi1000VIS(data, **kwargs):
    """
    NAME: BDI1000VIS
    PARAMETER: 1 micron integrated band depth; VIS wavelengths
    FORMULATION: divide R830, R860, R890, R915 by RPEAK1 then\
      integrate over (1 -  normalized radiances)
    RATIONALE: crystalline Fe+2 or Fe+3 minerals
    """
    peak = rpeak1(data)
    wvs = data.wavelengths
    kernels = {833: 5,
               860: 5,
               890: 5,
               915: 5,
               peak:5}

    return generic_func(data, wvs, func=cf.bdi1000VIS_func, pass_wvs=True, kernels=kernels, **kwargs)

# TODO: bdi1000IR
def bdi1000IR(data, **kwargs):
    """
    NAME: BDI1000IR
    PARAMETER: 1 micron integrated band depth; IR wavelengths
    FORMULATION: divide R1030, R1050, R1080, R1150\
     by linear fit from peak R  between 1.3 - 1.87 microns to R2530\
     extrapolated backwards, then integrate over (1 -  normalized\
     radiances)
    RATIONALE: crystalline Fe+2 minerals; corrected for overlying\
    aerosol induced slope
    """
    wvs = data.wavelengths
    # medR A
    A_mask = (wvs >= 1300) * (wvs <= 1870)
    A_wvs = wvs[A_mask]
    medRA = np.median(A_wvs)

    # medR B
    B_mask = (wvs >= 2430) * (wvs <= 2600)
    B_wvs = wvs[B_mask]
    medRB = np.median(B_wvs)

    kernels = {1045: 5,
               1066: 5,
               1087: 5,
               1108: 5,
               1129: 5,
               1150: 5,
               1171: 5,
               1192: 5,
               1213: 5,
               1234: 5,
               1255: 5,
               medRA:15,
               medRB:15}

    return generic_func(data, wvs, func=cf.bdi1000IR_func, pass_wvs=True, kernels=kernels, **kwargs)

def r1300(data, **kwargs):
    """
    Name: R1300
    Parameter: 1.30 micron reflectance
    FORMULATION (with kernels): R1300
    Rationale: IRA browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1300]
    kernels = {1300: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def r1330(data, **kwargs):
    """
    NAME: R1330
    PARAMETER: IR albedo
    FORMULATION: R1330
    RATIONALE: IR albedo (ices > dust > unaltered mafics)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1330]
    kernels = {1330: 11}

    return generic_func(data, wv, func = (lambda x : x[0]), kernels = kernels, **kwargs)


def bd1300(data, **kwargs):
    '''
    NAME: BD1300
    PARAMETER: 1.3 μm absorption associated with Fe2+ substitution in\
        plagioclase
    FORMULATION (with kernels): 1 - ( R1320 / (a * R1080 + b * R1750) )
    RATIONALE: Plagioclase with Fe2+ substitution

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    '''

    wv = [1080, 1320, 1750]
    kernels = {1370: 5,
               1432: 15,
               1470: 5}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def olivine_index2(data, **kwargs):
    """
    NAME: OLINDEX (prior to TRDR version 3)
    PARAMETER: olivine index
    FORMULATION: (R1695 / (0.1*R1080 + 0.1*R1210 + 0.4*R1330 +\
      0.4*R1470)) - 1
    RATIONALE: olivine will be strongly +; based on fayalite

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    """

    wv = [1080, 1210, 1330, 1470, 1750, 2400]
    return generic_func(data, wv, func = cf.olivine_index2_func, **kwargs)

#@@TODO olivine_index2 (labeled olivine_index3 in JPL doc?
def olivine_index3(data, **kwargs):
    """
    NAME: OLINDEX2 (beginning with TRDR version 3)
    PARAMETER: olivine index with less sensitivity to illumination
    FORMULATION: RB1080 * 0.03 + RB1152 * 0.03 + RB1210 * 0.03 +\
        RB1250 * 0.03 + RB1263 * 0.07 + RB1276 * 0.07 +\
        RB1330 * 0.12 + RB1368 * 0.12 + RB1395 * 0.14 +\
        RB1427 * 0.18 + RB1470 * 0.18
    RATIONALE: olivine will be strongly positive
    """

    wv = [1080, 1152, 1210, 1250, 1263, 1276, 1330, 1368, 1395, 1427, 1470, 1750, 2400]
    kernels = {1080: 7, 1152: 7, 1210: 7, 1250: 7, 1263: 7, 1276: 7, 1330: 7, 1368: 7,
               1395: 7, 1427: 7, 1470: 7, 1750: 7, 2400: 7}

    return generic_func(data, wv, func = cf.olivine_index3_func, kernels = kernels, **kwargs)


def lcp_index(data, **kwargs):
    """
    NAME: LCPINDEX
    PARAMETER: LCP index
    FORMULATION: ((R1330 - R1050)/(R1330 + R1050)) *\
                   ((R1330 - R1815)/(R1330 + R1815))
    RATIONALE: Pyroxene is strongly +; favors LCP

    Algorithm differs from published - coded as per CAT <--- What?

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1080, 1330, 1815]
    return generic_func(data, wv, func = cf.index1_func, **kwargs)


def lcp_index2(data, **kwargs):
    """
    NAME: LCPINDEX2
    PARAMETER: Detect broad absorption centered at 1.81 μm
    FORMULATION (with kernels):\
        RB1690 * 0.20 + RB1750 * 0.20 + RB1810 * 0.30 + RB1870 * 0.30\
        Anchored at R1560 and R2450
    RATIONALE: Pyroxene is strongly +; favors LCP

    Algorithm differs from published - coded as per CAT <--- What?

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1690, 1750, 1810, 1870]
    kernels = {1560: 7,
               1690: 7,
               1750: 7,
               1810: 7,
               1870: 7,
               2450: 7}

    return generic_func(data, wv, func = cf.lcp_index2_func, kernels = kernels, **kwargs)


def hcp_index(data, **kwargs):
    """
    NAME: HCPXINDEX
    PARAMETER: pyroxene index
    FORMULATION: 100 * ((R1470 - R1050) / (R1470 + R1050)) *\
                         ((R1470 - R2067) / (R1470 + R2067))
    RATIONALE: pyroxene is strongly +; favors high-Ca pyroxene

    Algorithm differs from published - coded as per CAT <--- What?

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1050, 1470, 2067]
    return generic_func(data, wv, func = cf.index1_func, **kwargs)

def hcp_index2(data, **kwargs):
    """
    NAME: HCPXINDEX
    PARAMETER: pyroxene index
    FORMULATION: 100 * ((R1470 - R1050) / (R1470 + R1050)) *\
                         ((R1470 - R2067) / (R1470 + R2067))
    RATIONALE: pyroxene is strongly +; favors high-Ca pyroxene

    Algorithm differs from published - coded as per CAT <--- What?

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2140, 2230, 2250, 2430, 2460]
    kernels = {1810: 7,
               2120: 5,
               2140: 7,
               2230: 7,
               2250: 7,
               2430: 7,
               2460: 7,
               2530: 7}

    return generic_func(data, wv, func = cf.hcp_index2_func, kernels = kernels, **kwargs)

'''#@@TODO var
def var(data, **kwargs):
    """
    NAME: VAR
    PARAMETER: spectral variance
    FORMULATION: find variance from a line fit from 1 - 2.3 micron
      by summing in quadrature over the intervening wavelengths
    RATIONALE: Ol & Px will have high values; Type 2 areas will have
      low values
    """

    raise NotImplementedError'''


def islope1(data, **kwargs):
    """
    NAME: ISLOPE1
    PARAMETER: -1 * spectral slope1
    FORMULATION: (R1815-R2530) / (2530-1815)
    RATIONALE: ferric coating on dark rock

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1815, 2530]
    kernels = {1815: 5,
               2530: 5}

    return generic_func(data, wv, func = cf.islope1_func, kernels = kernels, pass_wvs = True, **kwargs)


def bd1400(data, **kwargs):
    """
    NAME: BD1400
    PARAMETER: 1.4 micron H2O and OH band depth
    FORMULATION: 1 - ( R1395 / (a * R1330 + b * R1467) )
    RATIONALE: Hydrated or hydroxylated minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    """

    wv = [1330, 1395, 1467]
    kernels = {1370: 5,
               1432: 3,
               1470: 5}
    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def bd1435(data, **kwargs):
    """
    NAME: BD1435
    PARAMETER: 1.435 micron band depth
    FORMULATION: 1 - ( R1435 / (a * R1370 + b * R1470) )
    RATIONALE: CO2 ice, some hydrated minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1370, 1435, 1470]
    kernels = {1370: 3,
               1432: 1,
               1470: 3}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def bd1500(data, use_kernels = True, **kwargs):
    """
    NAME: BD1500
    PARAMETER: 1.5 micron H2O ice band depth
    FORMULATION: 1.0 - ((R1505 + R1558) / (R1808 + R1367))
    FORMULATION (with kernels): 1.0 - (R1525 / (b * R1808 + a * R1367))
    RATIONALE: H2O surface ice
    Algorithm differs from published - coded as per CAT (reduced instrument noise)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    if use_kernels:
        wv = [1367, 1525, 1808]
        kernels = {1367: 5,
                   1525: 11,
                   1808: 5}

        return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

    wv = [1367, 1505, 1558, 1808]
    return generic_func(data, wv, func = cf.bd1500_func, **kwargs)


def icer1(data, **kwargs):
    """
    NAME: ICER1
    PARAMETER: 1.5 micron and 1.43 micron band ratio
    FORMULATION (with kernels): R1510 / R1430
    RATIONALE: CO2, H20 ice mixtures

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1430, 1510]
    kernels = {1430: 5,
               1510: 5}

    return generic_func(data, wv, func = cf.rockdust2_func, kernels = kernels, **kwargs)


def icer1_2(data, **kwargs):
    """
    NAME: ICER1_2
    PARAMETER: 1.5 micron and 1.43 micron band ratio
    FORMULATION: 1 - ((1 - bd1435) / (1 - bd1500))
    RATIONALE: CO2, H20 ice mixtures

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    bd1435_val = bd1435(data)
    bd1500_val = bd1500(data, use_kernels = False)

    return 1 - ((1 - bd1435_val) / (1 - bd1500_val))


def bd1750(data, use_kernels = True, **kwargs):
    """
    NAME: BD1750
    PARAMETER: 1.7 micron band depth
    FORMULATION: 1 - ( R1750 / (a * R1550 + b * R1815) )
    FORMULATION (with kernels): 1 - ( R1750 / (a * R1690 + b * R1815) )
    RATIONALE: gypsum

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    """

    wv = [1550, 1750, 1815]
    kernels = {}

    if use_kernels:
        wv = [1690, 1750, 1815]
        kernels[1690] = 5
        kernels[1750] = 3
        kernels[1815] = 5

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def bd1900(data, **kwargs):
    """
    NAME: BD1900
    PARAMETER: 1.9 micron band depth
    FORMULATION: 1.0 - ((R1985 + R1930)/(R2067 + R1875))
    RATIONALE: H2O, chemically bound or adsorbed
    Algorithm differs from published - coded as per CAT (reduced instrument noise)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1875, 1930, 1985, 2067]
    return generic_func(data, wv, func = cf.bd1900_func, pass_wvs = True, **kwargs)

def bd1900_2(data, **kwargs):
    """
    NAME: BD1900_2
    PARAMETER: 1.9 micron band depth
    FORMULATION (with kernels):\
        .5 * (1 - (R1930 / (a * R1850 + b * R2067))) +\
        .5 * (1 - (R1985 / (a * R1850 + b * R2067)))
    RATIONALE: H2O, chemically bound or adsorbed

    Algorithm differs from published - coded as per CAT (reduced instrument noise)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [1850, 1930, 2067]
    kernel_set1 = {1850: 5,
                   1930: 5,
                   2046: 5}

    wv_set2 = [1850, 1985, 2067]
    kernel_set2 = {1850: 5,
                   1985: 5,
                   2046: 5}

    bd_1 = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set1, **kwargs)
    bd_2 = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set2, **kwargs)

    return .5 * (1 - bd_1) + .5 * (1 - bd_2)

def bd1900r(data, **kwargs):
    """
    NAME: BD1900r
    PARAMETER: 1.9 micron band depth
    FORMULATION: 1.0 - ((R1908 + R1914 + R1921 + R1928 + R1934 + R1941) / \
                          (R1862 + R1869 + R1875 + R2112 + R2120 + R2126))
    RATIONALE: H2O, chemically bound or adsorbed
    Algorithm differs from published - coded as per CAT (reduced instrument noise)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1908, 1914, 1921, 1928, 1934, 1941,
          1862, 1869, 1875, 2112, 2120, 2126]

    return generic_func(data, wv, func = cf.bd1900r_func, **kwargs)

def bd1900r2(data, **kwargs):
    """
    NAME: BD1900r2
    PARAMETER: 1.9 micron band depth
    FORMULATION:\
1 - ((R1908 / RC1908 + R1914 / RC1914 + R1921 / RC1921 + R1928 / RC1928 + R1934 / RC1934 + R1941 / RC1941) /\
      (R1862 / RC1862 + R1869 / RC1869 + R1875 / RC1875 + R2112 / RC2112 + R2120 / RC2120 + R2126 / RC2126))
    RATIONALE: H2O, chemically bound or adsorbed

    Algorithm differs from published - coded as per CAT (reduced instrument noise)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1850, 1862, 1869, 1875, 1908, 1914, 1921, 1928, 1934, 1941, 2060, 2112, 2120, 2126]

    return generic_func(data, wv, func = cf.bd1900r2_func, pass_wvs = True, **kwargs)

def bdi2000(data, **kwargs):
    """
    NAME: BDI2000
    PARAMETER: 2 micron integrated band depth
    FORMULATION: divide R1660, R1815, R2140, R2210, R2250, R2290,\
      R2330, R2350, R2390, R2430, R2460 by linear fit from peak R\
      between 1.3 - 1.87 microns to R2530, then integrate over\
     (1 -  normalized radiances)
    RATIONALE: pyroxene abundance and particle size
    """
    wvs = data.wavelengths
    # peakR of 15
    peak_mask = (wvs >= 1300) * (wvs <= 1870)
    peak_wvs = wvs[peak_mask]
    peakR = generic_func(data, peak_wvs, func = cf.rpeak1_func, pass_wvs = True, **kwargs)
    
    kernels = {1660: 5,
               1815: 5,
               2140: 5,
               2210: 5,
               2250: 5,
               2290: 5,
               2330: 5,
               2350: 5,
               2390: 5,
               2430: 5,
               2460: 5,
               2530: 5,
               peakR:15}


    return generic_func(data, peak_wvs, func=cf.bdi2000_func, pass_wvs=True, kernels=kernels, **kwargs)

def bd2100(data, use_kernels = True, **kwargs):
    """
    NAME: BD2100
    PARAMETER: 2.1 micron band depth
    FORMULATION: 1 - ( ((R2120 + R2140) * 0.5) / (a * R1930 + b * R2250) )
    FORMULATION (with kernels): 1 - ( R2132 / (a * R1930 + b * R2250) )
    RATIONALE: monohydrated minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    if use_kernels:
        wv = [1930, 2132, 2250]
        kernels = {1930: 3,
                   2132: 5,
                   2250: 3}

        return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

    wv = [1930, 2120, 2130, 2250]
    return generic_func(data, wv, func = cf.bd2100_func, pass_wvs=True, **kwargs)


def bd2165(data, **kwargs):
    """
    NAME: BD2165
    PARAMETER: 2.165 micron Al-OH band depth
    FORMULATION (with kernels): 1 - ( R2165 / (a * R2120 + b * R2230) )
    RATIONALE: Pyrophyllite Kaolinite group

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2165, 2230]
    kernels = {2120: 5,
                      2165: 3,
                      2230: 3}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def bd2190(data, **kwargs):
    """
    NAME: BD2190
    PARAMETER: 2.190 micron Al-OH band depth
    FORMULATION (with kernels): 1 - ( R2185 / (a * R2120 + b * R2250) )
    RATIONALE: Beidellite Allophane Imogolite

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2185, 2250]
    kernels = {2120: 5,
                      2185: 3,
                      2250: 3}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def doub2200h(data, **kwargs):
    """
    NAME: DOUB2200H
    PARAMETER: 2.16 micron Si-OH band depth and 2.21 micron H-bound Si-OH band\
        depth (doublet)
    FORMULATION (with kernels): 1 - ((R2205 + R2258) / (R2172 + R2311))
    RATIONALE: Opal and other Al-OH minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2172, 2205, 2258, 2311]
    kernels = {2172: 5,
                      2205: 3,
                      2258: 3,
                      2311: 5}

    return generic_func(data, wv, func = cf.doub2200h_func, kernels = kernels, **kwargs)

def min2200(data, **kwargs):
    """
    NAME: MIN2200
    PARAMETER: 2.16 μm Si-OH band depth and 2.21 μm H-bound Si-OH band\
        depth (doublet)
    FORMULATION (with kernels): minimum( 1 -  (R2165 / (a * R2120 + b * R2350)),\
        1 - (R2210 / (a * R2120 + b * R2350)))
    RATIONALE: Kaolinite group

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [2120, 2165, 2350]
    kernel_set1 = {2120: 5, 2165: 3, 2350: 5}

    wv_set2 = [2120, 2210, 2350]
    kernel_set2 = {2120: 5, 2210: 3, 2350: 5}

    bd_1 = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set1, **kwargs)
    bd_2 = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set2, **kwargs)

    return np.minimum(bd_1, bd_2)

def bd2210(data, use_kernels = True, **kwargs):
    """
    NAME: BD2210
    PARAMETER: 2.21 micron band depth
    FORMULATION: 1 - ( R2210 / (a*R2140+b*R2250) )
    RATIONALE: Al-OH minerals: monohydrated minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2210, 2250]
    kernels = {}

    if use_kernels:
        wv = [2165, 2210, 2250]
        kernels[2165] = 5
        kernels[2210] = 5
        kernels[2250] = 5

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def d2200(data, **kwargs):
    """
    NAME: D2200
    PARAMETER: 2.2 micron dropoff
    FORMULATION (with kernels): 1 - (((R2210 / RC2210) + (R2230 / RC2230)) / (2 * (R2165 / RC2165)))\
        Slope for RC#### anchored at R1815 and R2430.
    RATIONALE: Al-OH minerals

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1815, 2165, 2210, 2230, 2430]
    kernels = {1815: 7,
                      2165: 5,
                      2210: 7,
                      2230: 7,
                      2430: 7}

    return generic_func(data, wv, func = cf.d2200_func, kernels = kernels, **kwargs)

def bd2230(data, **kwargs):
    """
    NAME: BD2230
    PARAMETER: 2.23 μm band depth
    FORMULATION (with kernels): 1 - (R2235 / (a * R2210 + b * R2252))
    RATIONALE: Hydroxylated ferric sulfates

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2210, 2235, 2252]
    kernels = {2210: 3,
                      2235: 3,
                      2252: 3}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def bd2250(data, **kwargs):
    """
    NAME: BD2250
    PARAMETER: 2.25 μm band depth
    FORMULATION (with kernels): 1 - (R2245 / (a * R2120 + b * R2340))
    RATIONALE: 2.25 μm broad Al-OH and Si-OH band depth

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2245, 2340]
    kernels = {2120: 5,
                      2245: 7,
                      2340: 3}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def min2250(data, **kwargs):
    """
    NAME: MIN2250
    PARAMETER: 2.21 μm Si-OH band depth and 2.26 μm H-bound Si-OH\
        band depth
    FORMULATION (with kernels): minimum( 1 -  (R2210 / (a * R2165 + b * R2350)),\
        1 - (R2265 / (a * R2165 + b * R2350)))
    RATIONALE: Opal

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [2165, 2210, 2350]
    kernel_set1 = {2165: 5, 2210: 3, 2350: 5}

    wv_set2 = [2165, 2265, 2350]
    kernel_set2 = {2165: 5, 2265: 5, 2350: 5}

    bd_1 = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set1, **kwargs)
    bd_2 = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set2, **kwargs)

    return np.minimum(bd_1, bd_2)

def bd2265(data, **kwargs):
    """
    NAME: BD2265
    PARAMETER: 2.265 micron band depth
    FORMULATION: 1 - ( R2265 / (a*R2210+b*R2340) )
    RATIONALE: Jarosite Gibbsite Acid-leached nontronite

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2210, 2265, 2340]
    kernels = {2210: 5, 2265: 3, 2340: 5}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def bd2290(data, **kwargs):
    """
    NAME: BD2290
    PARAMETER: 2.29 micron band depth
    FORMULATION: 1 - ( R2290 / (a*R2250+b*R2350) )
    RATIONALE: Mg,Fe-OH minerals (at 2.3); also CO2 ice

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2250, 2290, 2350]
    kernels = {2250: 5,
                      2290: 5,
                      2350: 5}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)


def d2300(data, **kwargs):
    """
    NAME: D2300
    PARAMETER: 2.3 micron drop
    FORMULATION: 1 - ( (CR2290+CR2320+CR2330) /\
      (CR2140+CR2170+CR2210) ) (CR values are observed R values\
      divided by values fit along the slope as determined between 1.8\
      and 2.53 microns - essentially continuum corrected))
    RATIONALE: hydrated minerals; particularly clays

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    """

    wv = [1815, 2120, 2170, 2210, 2290, 2320, 2330, 2530]
    kernels = {1815: 5,
                      2120: 5,
                      2170: 5,
                      2210: 5,
                      2290: 3,
                      2320: 3,
                      2330: 3,
                      2530: 5}

    return generic_func(data, wv, func = cf.d2300_func, kernels = kernels, **kwargs)

def bd2355(data, **kwargs):
    """
    NAME: BD2355
    PARAMETER: 2.35 micron band depth
    FORMULATION: 1 - ( R2355 / (a * R2300+b * R2450) )
    RATIONALE: Chlorite Prehnite Pumpellyite

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2300, 2355, 2450]
    kernels = {2300: 5,
                      2355: 5,
                      2450: 5}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels =kernels, **kwargs)

def sindex(data, **kwargs):
    """
    NAME: SINDEX
    PARAMETER: Convexity at 2.29 microns  due to absorptions at\
      1.9/2.1 microns and 2.4 microns
    FORMULATION: 1 - (R2100 + R2400) / (2 * R2290) CR\
      values are observed R values divided by values fit along the\
      slope as determined between 1.8 - 2.53 microns (essentially\
      continuum corrected))
    RATIONALE: hydrated minerals; particularly sulfates

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2100, 2400, 2290]

    return generic_func(data, wv, func = cf.sindex_func, **kwargs)

def sindex2(data, **kwargs):
    """
    NAME: SINDEX2
    PARAMETER: Inverse lever rule to detect convexity at 2.29 μm due to 2.1 μm\
        and 2.4 μm absorptions
    FORMULATION (with kernels): 1 - (a * R2120 + b * R2400) / R2290
    RATIONALE: Hydrated sulfates (mono and polyhydrated sulfates) will be\
        strongly > 0

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2120, 2290, 2400]
    kernels = {2120: 5, 2290: 7, 2400: 3}

    return generic_func(data, wv, func = cf.sh_func, pass_wvs = True, kernels = kernels, **kwargs)

def icer2(data, **kwargs):
    """
    NAME: ICER2
    PARAMETER: gauge 2.7 micron band
    FORMULATION: R2530 / R2600
    RATIONALE: CO2 ice will be >>1, H2O ice and soil will be about 1

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2530, 2600]

    return generic_func(data, wv, func = cf.rockdust2_inverse_func, **kwargs)

def bdcarb(data, **kwargs):
    """
    NAME: BDCARB
    PARAMETER: overtone band depth
    FORMULATION: 1 - ( sqrt [ ( R2330 / (a*R2230+b*R2390) ) *\
      ( R2530/(c*R2390+d*R2600) ) ] )
    RATIONALE: carbonate overtones

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [2230, 2330, 2390]
    wv_set2 = [2390, 2530, 2600]

    bd_1 = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, **kwargs)
    bd_2 = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, **kwargs)

    return 1 - np.sqrt((bd_1 * bd_2))

def min2295_2480(data, **kwargs):
    """
    NAME: MIN2295_2480
    PARAMETER: Mg Carbonate overtone band depth and metal-OH band
    FORMULATION (with kernels): minimum( 1 -  (R2295 / (a * R2165 + b * R2364)),\
        1 - (R2480 / (a * R2364 + b * R2570)))
    RATIONALE: Mg carbonates; both overtones must be present

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [2165, 2295, 2364]
    kernel_set1 = {2165: 5, 2295: 5, 2364: 5}

    wv_set2 = [2364, 2480, 2570]
    kernel_set2 = {2364: 5, 2480: 5, 2570: 5}

    bd2295_val = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set1, **kwargs)
    bd2480_val = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set2, **kwargs)

    return np.minimum(bd2295_val, bd2480_val)

def min2345_2537(data, **kwargs):
    """
    NAME: MIN2345_2537
    PARAMETER: Ca/Fe Carbonate overtone band depth and metal-OH band
    FORMULATION (with kernels): minimum( 1 -  (R2345 / (a * R2250 + b * R2430)),\
        1 - (R2537 / (a * R2430 + b * R2602)))
    RATIONALE: Ca/Fe carbonates; both overtones must be present

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv_set1 = [2250, 2345, 2430]
    kernel_set1 = {2250: 5, 2345: 5, 2430: 5}

    wv_set2 = [2430, 2537, 2602]
    kernel_set2 = {2430: 5, 2537: 5, 2602: 5}

    bd2345_val = generic_func(data, wv_set1, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set1, **kwargs)
    bd2537_val = generic_func(data, wv_set2, func = cf.bd_func2, pass_wvs = True, kernels = kernel_set2, **kwargs)

    return np.minimum(bd2345_val, bd2537_val)

def bd2500h(data, use_kernels = True, **kwargs):
    """
    NAME: BD2500h
    PARAMETER: Mg Carbonate overtone band depth
    FORMULATION: 1 - ((R2500 + R2510) /  (R2540 + R2380))
    FORMULATION (with kernels): 1 - (R2480 / ((a * R2364) + (b * R2570)))
    RATIONALE: Mg carbonates

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2380, 2500, 2510, 2540]

    if use_kernels:
        wv = [2364, 2480, 2570]
        kernels = {2364: 5, 2480: 5, 2570: 5}

        return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

    return generic_func(data, wv, func = cf.bd2500h_func, **kwargs)

def bd3000(data, **kwargs):
    """
    NAME: BD3000
    PARAMETER: 3 micron band depth
    FORMULATION: 1 - ( R3000 / (R2530*(R2530/R2210)) )
    RATIONALE: H2O, chemically bound or adsorbed

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2210, 2530, 3000]
    kernels = {2210: 5, 2530: 5, 3000: 5}

    return generic_func(data, wv, func = cf.bd3000_func, kernels = kernels, **kwargs)

def bd3100(data, **kwargs):
    """
    NAME: BD3100
    PARAMETER: 3.1 micron band depth
    FORMULATION: 1 - ( R3120 / (a*R3000+b*R3250) )
    RATIONALE: H2O ice

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [3000, 3120, 3250]

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, **kwargs)

def bd3200(data, **kwargs):
    """
    NAME: BD3200
    PARAMETER: 3.2 micron band depth
    FORMULATION: 1 - ( R3320 / (a*R3250+b*R3390) )
    RATIONALE: CO2 ice

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [3250, 3320, 3390]

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, **kwargs)

def bd3400(data, use_kernels = True, **kwargs):
    """
    NAME: BD3400
    PARAMETER: 3.4 micron band depth
    FORMULATION: 1 - ( (a*R3390+b*R3500) / (c*R3250+d*R3630) )
    RATIONALE: carbonates; organics

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [3250, 3390, 3500, 3630]

    if use_kernels:
        wv = [3250, 3320, 3390]
        kernels = {3250: 10, 3420: 15, 3630: 10}

        return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

    return generic_func(data, wv, func = cf.bd3400_func, pass_wvs = True, **kwargs)

def cindex(data, **kwargs):
    """
    NAME: CINDEX
    PARAMETER: gauge 3.9 micron band
    FORMULATION: ( R3750 + (R3750-R3630) / (3750-3630) *\
      (3920-3750) ) / R3950 - 1
    RATIONALE: carbonates

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    Algorithm differs from published - coded as per CAT
    """

    wv = [3630, 3750, 3950]

    return generic_func(data, wv, func = cf.cindex_func, **kwargs)

def cindex2(data, **kwargs):
    """
    NAME: CINDEX
    PARAMETER: Inverse lever rule to detect convexity at 3.6 μm due to 3.4 μm and 3.9 μm\
        absorptions
    FORMULATION (with kernels): 1 - ((a * R3450 +  b * R3875) / 3610)
    RATIONALE: carbonates

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray

    Algorithm differs from published - coded as per CAT
    """

    wv = [3450, 3875, 3610]
    kernels = {3450: 9, 3875: 11, 3610: 7}

    return generic_func(data, wv, func = cf.sh_func, pass_wvs = True, kernels = kernels, **kwargs)

def r440(data, **kwargs):
    """
    Name: R440
    Parameter: 0.44 micron reflectance
    FORMULATION (with kernels): R440
    Rationale: Clouds/Hazes

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [440]
    kernels = {440: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def r530(data, **kwargs):
    """
    Name: R530
    Parameter: 0.53 micron reflectance
    FORMULATION (with kernels): R530
    Rationale: TRU browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [530]
    kernels = {530: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def r600(data, **kwargs):
    """
    Name: R600
    Parameter: 0.60 micron reflectance
    FORMULATION (with kernels): R600
    Rationale: TRU browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [600]
    kernels = {600: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def irr1(data, **kwargs):
    """
    Name: IRR1
    Parameter: IR ratio 1
    FORMULATION (with kernels): R800 / R997
    Rationale: Aphelion ice clouds (>1) versus seasonal or

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [800, 997]
    kernels  = {800: 5, 997: 5}

    return generic_func(data, wv, func = cf.rockdust2_inverse_func, kernels = kernels, **kwargs)

def r1080(data, **kwargs):
    """
    Name: R1080
    Parameter: 1.08 micron reflectance
    FORMULATION (with kernels): R1080
    Rationale: FAL browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1080]
    kernels = {1080: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)


def r1506(data, **kwargs):
    """
    Name: R1506
    Parameter: 1.51 micron reflectance
    FORMULATION (with kernels): R1506
    Rationale: TRU browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [1506]
    kernels = {1506: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def r2529(data, **kwargs):
    """
    Name: R2529
    Parameter: 2.53 micron reflectance
    Formulation: R2529
    Rationale: TRU browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2529]
    kernels = {2529: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)

def bd2600(data, **kwargs):
    """
    NAME: BD2600
    PARAMETER: 2.6 μm H 2 O band depth
    FORMULATION: 1 - (R2600 / (a * R2530 + b * R2630))
    RATIONALE: H 2 O vapor (accounts for spectral slope)

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2530, 2600, 2630]
    kernels = {2530: 5, 2600: 5, 2630: 5}

    return generic_func(data, wv, func = cf.bd_func2, pass_wvs = True, kernels = kernels, **kwargs)

def irr2(data, **kwargs):
    """
    Name: IRR2
    Parameter: IR ratio 2
    FORMULATION (with kernels): R2530 / R2210
    Rationale: Aphelion ice clouds versus seasonal or dust

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [2210, 2530]
    kernels = {2210: 5, 2350: 5}

    return generic_func(data, wv, func = cf.rockdust2_func, kernels = kernels, **kwargs)

def irr3(data, **kwargs):
    """
    Name: IRR3
    Parameter: IR ratio 3
    FORMULATION (with kernels): R3500 / R3390
    Rationale: Aphelion ice clouds (higher values) versus seasonal or dust

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [3390, 3500]
    kernels = {3390: 7, 3500: 7}

    return generic_func(data, wv, func = cf.rockdust2_func, kernels = kernels, **kwargs)

def r3920(data, **kwargs):
    """
    Name: R3920
    Parameter: 3.92 micron reflectance
    Formulation: R3920
    Rationale: IC2 browse product component

    Parameters
    ----------
    data : ndarray
           (n,m,p) array

    Returns
    -------
     : ndarray
       the processed ndarray
    """

    wv = [3920]
    kernels = {3920: 5}

    return generic_func(data, wv, func = (lambda x: x[0]), kernels = kernels, **kwargs)
