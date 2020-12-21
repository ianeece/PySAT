from math import e
import numpy as np

# Generic Function for all RXXX formulas
def reflectance_func(bands):
    return bands[0]

def visnir_func(bands):
    R699, R1579 = bands

    return R699 / R1579

def r950_750_func(bands):
    R749, R949 = bands

    return R949 / R749

# Generic Function for all BDXXX formulas
def bd_func(bands, wvs):
    return 1 - (bands[1] / ((bands[2] - bands[0]) / (wvs[2] - wvs[0]) * (wvs[1] - wvs[0]) + bands[0]))

def bdi_func(bands, wvs=[0,0]):
    lower_array, band_array, upper_array = bands
    lower_bound, y, upper_bound = wvs

    return band_array / (((upper_array - lower_array)/ \
           (upper_bound - lower_bound)) * (y - lower_bound) + lower_array)

def twoum_ratio_func(bands):
    R1578, R2538 = bands

    return R1578 / R2538

def thermal_ratio_func(bands):
    R2538, R2978 = bands

    return R2538 / R2978

def bd3000_func(bands):
    R1578, R2538, R2978 = bands

    return 1 - ((R2978) / (((R2538 - R1578) / (2538 - 1578)) * (2978 - 1578) + R1578))

def visslope_func(bands):
    R419, R749 = bands

    return (R749 - R419) / (749-419)

def oneum_slope_func(bands):
    R699, R1579 = bands

    return (R1579 - R699) / (1579 - 699)

def olindex_func(bands):
    R650, R860, R1047, R1230, R1750 = bands

    return 10 * (0.1 * ((((R1750 - R650) / (1750 - 650)) * (860 - 650) + R650) / R860)) + \
            (0.5 * ((((R1750 - R650) / (1750 - 650)) * (1047 - 650) + R650) / R1047)) + \
            (0.25 * ((((R1750 - R650) / (1230 - 650)) * (860 - 650) + R650) / R1230))

def twoum_slope_func(bands):
    R1578, R2538 = bands

    return (R2538 - R1578) / (2538 - 1578)
