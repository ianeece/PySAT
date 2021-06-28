import numpy as np
import pandas as pd
from libpyhat.examples import get_path
import libpyhat.regression.sm as sm
import libpyhat.regression.regression as reg

def test_sm_blend():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    predictions = [df[('predict','low')],df[('predict','mid')],df[('predict','high')],df[('predict','full')]]

    blendranges = [[-9999,30],[20,60],[50,9999]]
    sm_obj = sm.sm(blendranges, random_seed=0)
    blended_predictions = sm_obj.do_blend(np.array(predictions)) #without optimization
    rmse = np.sqrt(np.average((blended_predictions - df[('comp', 'SiO2')]) ** 2))
    np.testing.assert_almost_equal(rmse, 12.693586299818989, decimal=5)

    #re-initialize sm_obj to make sure it behaves
    blendranges = [[-9999, 30], [20, 60], [50, 9999]]
    sm_obj = sm.sm(blendranges, random_seed=0)
    blended_predictions = sm_obj.do_blend(np.array(predictions),truevals=np.array(df[('comp','SiO2')]), verbose=False) #with optimization
    rmse = np.sqrt(np.average((blended_predictions-df[('comp','SiO2')])**2))
    expected_blendranges = [-9999., 24.0502605, 31.98075128, 53.9562582, 65.98663281, 9999.]
    np.testing.assert_almost_equal(rmse, 12.578145384007286, decimal=5)
    np.testing.assert_allclose(expected_blendranges,np.sort(sm_obj.blendranges),rtol=1e-5)
