import pytest
import numpy as np

from libpyhat.derived.m3 import pipe

def test_bd620(m3_img):
    res = pipe.bd620(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[0.16030534, 0.14788732, 0.1372549 ],
                                                        [0.12804878, 0.12, 0.11290323],
                                                        [0.10659898, 0.10096154, 0.09589041]]))

def test_bd950(m3_img):
    res = pipe.bd950(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.873589, -0.735741, -0.635468],
                                                        [-0.559249, -0.499355, -0.451049],
                                                        [-0.411265, -0.37793 , -0.349593]]))

def test_bd1050(m3_img):
    res = pipe.bd1050(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.332263, -0.293201, -0.262357],
                                                        [-0.237385, -0.216754, -0.199422],
                                                        [-0.184657, -0.171927, -0.160839]]))

def test_bd1250(m3_img):
    res = pipe.bd1250(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[ 0.155646,  0.143527,  0.133159],
                                                        [ 0.124188,  0.11635 ,  0.109442],
                                                        [ 0.103309,  0.097826,  0.092896]]))

def test_bd1900(m3_img):
    res = pipe.bd1900(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.09989909, -0.09, -0.08188586],
                                                        [-0.07511381, -0.06937631, -0.06445312],
                                                        [-0.06018237, -0.05644242, -0.0531401 ]]))

def test_bd2300(m3_img):
    res = pipe.bd2300(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[0.28366762,  0.26470588,  0.2481203 ],
                                                        [0.23349057,  0.22048998,  0.20886076],
                                                        [0.19839679,  0.1889313,   0.18032787]]))
def test_bd2800(m3_img):
    res = pipe.bd2800(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.143911, -0.129139, -0.117117],
                                                        [-0.107143, -0.098734, -0.091549],
                                                        [-0.085339, -0.079918, -0.075145]]))

def test_bd3000(m3_img):
    res = pipe.bd3000(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.34513274, -0.32231405, -0.30232558],
                                                        [-0.28467153, -0.26896552, -0.25490196],
                                                        [-0.24223602, -0.23076923, -0.22033898]]))

def test_r1580(m3_img):
    res = pipe.r1580(m3_img)
    np.testing.assert_array_almost_equal(res, np.arange(1,10).reshape((3,3)))

def test_r950_750(m3_img):
    res = pipe.r950_750(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[10., 5.5, 4.],
                                                        [3.25, 2.8, 2.5],
                                                        [2.28571429, 2.125, 2.]]))

def test_olindex(m3_img):
    res = pipe.olindex(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[1.28081027, 1.31103736, 1.33712718],
                                                        [1.35993684, 1.38009259, 1.39806426],
                                                        [1.41421237, 1.42881891, 1.44210815]]))

def test_oneum_slope(m3_img):
    res = pipe.oneum_slope(m3_img)
    delta = np.abs(np.sum(res - 0.01022727))
    assert delta <= 1e-6

def test_r750(m3_img):
    res = pipe.r750(m3_img)
    np.testing.assert_array_almost_equal(res, np.arange(1,10).reshape((3,3)))

def test_r1580(m3_img):
    res = pipe.r1580(m3_img)
    np.testing.assert_array_almost_equal(res, np.arange(1,10).reshape((3,3)))

def test_r540(m3_img):
    res = pipe.r540(m3_img)
    np.testing.assert_array_almost_equal(res, np.arange(1,10).reshape((3,3)))

def test_r2780(m3_img):
    res = pipe.r2780(m3_img)
    np.testing.assert_array_almost_equal(res, np.arange(1,10).reshape((3,3)))

def test_twoum_ratio(m3_img):
    res = pipe.twoum_ratio(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[0.1, 0.18181818, 0.25],
                                                        [0.30769231, 0.35714286, 0.4],
                                                        [0.4375, 0.47058824, 0.5]]))

def test_thermal_ratio(m3_img):
    res = pipe.thermal_ratio(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[0.1, 0.18181818, 0.25],
                                                        [0.30769231, 0.35714286, 0.4],
                                                        [0.4375, 0.47058824, 0.5]]))
def test_twoum_slope(m3_img):
    res = pipe.twoum_slope(m3_img)
    delta = np.abs(np.sum(res - 0.009375))
    assert delta <= 1e-6

def test_visslope(m3_img):
    res = pipe.visslope(m3_img)
    delta = np.abs(np.sum(res - 0.02727273))
    assert delta <= 1e-6

def test_visnir(m3_img):
    res = pipe.visnir(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[ 0.1, 0.18181818, 0.25],
                                                        [0.30769231, 0.35714286, 0.4],
                                                        [0.4375, 0.47058824, 0.5]]))

def test_bdi1000(m3_img):
    res = pipe.bdi1000(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.358333, -0.325318, -0.297874],
                                                        [-0.2747  , -0.254872, -0.237713],
                                                        [-0.222719, -0.209504, -0.19777 ]]))

def test_bdi2000(m3_img):
    res = pipe.bdi2000(m3_img)
    np.testing.assert_array_almost_equal(res, np.array([[-0.026842, -0.024199, -0.022029],
                                                        [-0.020215, -0.018677, -0.017356],
                                                        [-0.016209, -0.015204, -0.014316]]))
