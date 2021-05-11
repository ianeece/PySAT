import numpy as np
import pandas as pd
from libpyhat.examples import get_path
import libpyhat.regression.local_regression as local_regression
np.random.seed(1)


df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
def test_local_regression():
    params = {'fit_intercept': True,
              'positive': False,
              'random_state': 1,
              'tol': 1e-2,
              'max_iter': 2000,
              'selection': 'random'
              }
    model = local_regression.LocalRegression(params, n_neighbors=10, verbose=True)
    predictions, coeffs, intercepts = model.fit_predict(df['wvl'],df[('comp','SiO2')],df['wvl'])
    expected_rmse = 3.6651844609454796
    expected_coefs = [ 0.06578675, 0.086692, 0.04730609, 0.02218215, -0.0043394]
    expected_intercepts = [91.78926076797339, 20.04199177586427, 24.5343744038724, 44.61417574720293, 70.67902610266518]

    rmse = np.sqrt(np.average((predictions - df[('comp','SiO2')])**2))
    np.testing.assert_almost_equal(rmse,expected_rmse)
    assert np.array(coeffs).shape == (103,44)
    np.testing.assert_array_almost_equal(expected_coefs, np.array(coeffs)[10,5:10])
    assert np.array(intercepts).shape[0] == 103
    np.testing.assert_array_almost_equal(intercepts[0:5], expected_intercepts)

def test_local_parallel():
    params = {'fit_intercept': True,
              'positive': False,
              'random_state': 1,
              'tol': 1e-2,
              'max_iter': 2000,
              'selection': 'random'
              }
    model = local_regression.LocalRegression(params, n_neighbors=10, verbose=True)
    model.neighbors.fit(df['wvl'])
    predictions, coeffs, intercepts = local_regression.fit_predict_parallel(0, x_train=df['wvl'], y_train=df[('comp','SiO2')], x_predict=df['wvl'], model=model.model,
                         neighbors=model.neighbors, verbose=True)
    expected_prediction = 68.91056448621582
    expected_coeffs = [-0., -0.01101594, -0.0231962, -0., -0.]
    expected_intercept = 91.78926076797339
    np.testing.assert_almost_equal(predictions, expected_prediction)
    np.testing.assert_array_almost_equal(coeffs[0:5],expected_coeffs)
    np.testing.assert_almost_equal(intercepts,expected_intercept)