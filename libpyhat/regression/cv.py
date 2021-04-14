# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:16:11 2016
This function is used to run cross validation
to help choose the optimal number of components. Folds are stratified
according to a user-specified column

@author: rbanderson
"""
import warnings
import time
import numpy as np
# from sklearn.linear_model import RANSACRegressor as RANSAC
import pandas as pd
from libpyhat.regression.regression import regression
from libpyhat.regression import local_regression
from sklearn.linear_model import enet_path, lasso_path
from sklearn.linear_model._base import _pre_fit
from sklearn.utils.validation import check_X_y, check_array
from sklearn import decomposition
import multiprocessing
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
import time
import copy
from sklearn.model_selection import LeaveOneGroupOut

def RMSE(ypred, y):
    return np.sqrt(np.mean((np.squeeze(ypred) - np.squeeze(y)) ** 2))

# def path_calc(X, y, X_holdout, y_holdout, alphas, paramgrid, colname = 'CV', yname = '', method = 'Elastic Net'):
#     #make a copy of the parameters before popping things off
#     copy_params = copy.deepcopy(paramgrid)
#     fit_intercept = copy_params.pop('fit_intercept')
#     precompute = copy_params.pop('precompute')
#     copy_X = copy_params.pop('copy_X')
#     normalize = False

#     # this code adapted from sklearn ElasticNet fit function, which unfortunately doesn't accept multiple alphas at once
#     X, y = check_X_y(X, y, accept_sparse='csc',
#                      order='F', dtype=[np.float64, np.float32],
#                      copy=copy_X and fit_intercept,
#                      multi_output=True, y_numeric=True)
#     y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
#                     ensure_2d=False)

#     #this is the step that gives the data to find intercept if fit_intercept is true.
#     X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(X, y, None, precompute, normalize,
#                                                                  fit_intercept, copy=False)
#     y = np.squeeze(y)

#     #do the path calculation, and tell how long it took
#     print('Calculating path...')
#     start_t = time.time()
#     if method == 'Elastic Net':
#         path_alphas, path_coefs, path_gaps, path_iters = enet_path(X, y, alphas=alphas, return_n_iter = True,
#                                                    **copy_params)
#     if method == 'LASSO':
#         path_alphas, path_coefs, path_gaps, path_iters = lasso_path(X, y, alphas=alphas, return_n_iter=True,
#                                                                    **copy_params)
#     dt = time.time() - start_t
#     print('Took ' + str(round(dt,2)) + ' seconds')

#     #create some empty arrays to store the result
#     y_pred_holdouts = np.empty(shape=(len(alphas),len(y_holdout)))
#     intercepts = np.empty(shape=(len(alphas)))
#     rmses = np.empty(shape=(len(alphas)))
#     cvcols = []
#     for j in list(range(len(path_alphas))):

#         coef_temp = path_coefs[:, j]

#         if fit_intercept:
#             coef_temp = coef_temp / X_scale
#             intercept = y_offset - np.dot(X_offset, coef_temp.T)
#         else:
#             intercept = 0.

#         y_pred_holdouts[j,:] = np.dot(X_holdout, path_coefs[:, j]) + intercept
#         intercepts[j] = intercept
#         rmses[j] = RMSE(y_pred_holdouts[j,:], y_holdout)
#         cvcols.append(('predict','"'+ method + ' - ' + yname + ' - ' + colname + ' - Alpha:' + str(path_alphas[j]) + ' - ' + str(paramgrid) + '"'))

#     return path_alphas, path_coefs, intercepts, path_iters, y_pred_holdouts, rmses, cvcols

def cv_core(i, paramgrid = None, Train = None, xcols='wvl', ycol=('comp', 'SiO2'), method='PLS',
              yrange=None, calc_path = False, alphas = None):

    print('Permutation ' + str(i + 1) + ' of ' + str(len(paramgrid)))
    paramstring = ''
    for key in paramgrid[i].keys():
        paramstring = paramstring + key + ': ' + str(paramgrid[i][key]) + '; '
    print(paramstring[:-2])

    try:
        # create an iterator for cross validation based on the predefined folds
        cv_iterator = LeaveOneGroupOut().split(Train[xcols], Train[ycol], Train[('meta', 'Folds')])
        n_folds = LeaveOneGroupOut().get_n_splits(groups=Train[('meta', 'Folds')])

    except KeyError:
        print('***No folds found! Did you remember to define folds before running cross validation?***')
        return 0

    # create an empty output data frame to serve as template
    output_tmp = pd.DataFrame()
    # add columns for RMSEC, RMSECV, and RMSE for the folds
    output_tmp['RMSEC'] = 0
    output_tmp['RMSECV'] = 0

    # for f in np.array(range(n_folds)) + 1:
    for f in np.array(range(n_folds)) + 1:
        output_tmp['Fold ' + str(f)] = 0
    # fill in the output template based on the current permutation parameters
    for k in paramgrid[i].keys():
        output_tmp.at[0, k] = paramgrid[i][k]
    # if alphas is not None:  #this was used for path calc. comment out until that can be debugged
    #     output_tmp = pd.concat([output_tmp]*len(alphas))
    #     output_tmp['alphas'] = alphas

    output_tmp['Method'] = method

    rmsecv_folds_tmp = np.empty(shape=(0))  # Create empty array to hold RMSECV for each fold
    alphas_out = np.empty(shape=(0))
    cvcols_all = np.empty(shape=(0))
    cvcols = None
    foldcount = 1

    for train, holdout in cv_iterator:  # Iterate through each of the folds in the training set

        cv_train = Train.iloc[train]  # extract the data to be used to create the model
        cv_holdout = Train.iloc[holdout]  # extract the data that will be held out of the model

        if calc_path:
            print('Path calculation implementation is buggy and is currently disabled')

        #                     # get X and y data
        #                     X = cv_train[xcols]
        #                     y = cv_train[ycol]

        #                     #do the path calculation
        #                     path_alphas,\
        #                     path_coefs,\
        #                     intercepts,\
        #                     path_n_iters,\
        #                     y_pred_holdouts,\
        #                     fold_rmses,\
        #                     cvcols = path_calc(X, y, cv_holdout[xcols], cv_holdout[ycol], alphas, self.paramgrid[i], yname = ycol[0][-1], method = method)

        #                     output_tmp['Fold '+str(foldcount)] = fold_rmses
        #                     for n in list(range(len(path_alphas))):
        #                         Train.at[Train.index[holdout], cvcols[n]] = y_pred_holdouts[n]

        else:

            if method == 'Local Regression':
                params = paramgrid[i]
                try:
                    # on the first pass, pop off the n_neigbors parameter so it can be passed correctly
                    n_neighbors = params['n_neighbors']
                    params.pop('n_neighbors')
                    verbose = params['verbose']
                    params.pop('verbose')
                except:
                    pass
                cvcols = [('predict', '"' + method + '- CV -' + str(paramgrid[i]) + ' n_neighbors: ' + str(
                    n_neighbors) + '"')]
                model = local_regression.LocalRegression(params, n_neighbors=n_neighbors, verbose=verbose)
                y_pred_holdout, coeffs, intercepts = model.fit_predict(cv_train[xcols], cv_train[ycol],
                                                                       cv_holdout[xcols])
            else:
                cvcols = [('predict', '"' + method + '- CV -' + str(paramgrid[i]) + '"')]

                # fit the model and predict the held-out data
                model = regression([method], [paramgrid[i]])
                model.fit(cv_train[xcols], cv_train[ycol])
                if model.goodfit:
                    y_pred_holdout = model.predict(cv_holdout[xcols])
                else:
                    y_pred_holdout = cv_holdout[ycol] * np.nan
            # add the predictions to the appropriate column in the training data
            Train.at[Train.index[holdout], cvcols[0]] = y_pred_holdout
            # append the RMSECV to the list
            output_tmp['Fold ' + str(foldcount)] = RMSE(y_pred_holdout, cv_holdout[ycol])
            pass

        foldcount = foldcount + 1

    # now that all the folds have been held out and predicted, calculate the overall rmsecv and add it to the output
    rmsecv = []
    if cvcols is not None:
        for col in cvcols:
            rmsecv.append(RMSE(Train[col], Train[ycol]))
            predictkeys = [col[-1]]
        output_tmp['RMSECV'] = rmsecv

    # fit the model on the full training set using the current settings
    if calc_path:
        pass
        # X = Train[xcols]
        # y = Train[ycol]
        #
        # path_alphas, \
        # path_coefs, \
        # intercepts, \
        # path_n_iters, \
        # ypred_train, \
        # rmsec_train, \
        # cols = path_calc(X, y, X, y, alphas, self.paramgrid[i], colname = 'Cal', yname = ycol[0][-1], method = method)
        #
        #
        # for n in list(range(len(path_alphas))):
        #     Train[cols[n]]=ypred_train[n] #put the training set predictions in the data frame
        #     predictkeys.append(cols[n][-1])
        #     #create the model and manually set its parameters based on the path results rather than training it
        #     model = regression([method], [self.paramgrid[i]])
        #     model.model.set_params(alpha = path_alphas[n])
        #     setattr(model.model, 'intercept_', intercepts[n])
        #     setattr(model.model, 'coef_', np.squeeze(path_coefs)[:,n])
        #     setattr(model.model, 'n_iter_', path_n_iters[n])
        #
        #     #add the model and its name to the list
        #     models.append(model)
        #     modelkey = "{} - {} - ({}, {}) Alpha: {}, {}".format(method, ycol[0][-1], yrange[0], yrange[1],path_alphas[n],
        #                                               self.paramgrid[i])
        #     modelkeys.append(modelkey)
        #
        # output_tmp['RMSEC'] = rmsec_train
    else:
        if method == 'Local Regression':
            model = local_regression.LocalRegression(paramgrid[i], n_neighbors=n_neighbors,
                                                     verbose=verbose)
            modelkey = "{} - {} - ({}, {}) {} n_neighbors: {}".format(method, ycol[-1], yrange[0], yrange[1],
                                                                      paramgrid[i], n_neighbors)
        else:
            model = regression([method], [paramgrid[i]])
            modelkey = "{} - {} - ({}, {}) {}".format(method, ycol[-1], yrange[0], yrange[1],
                                                      paramgrid[i])

        ypred_train = Train[ycol] * np.nan
        if method == 'Local Regression':
            ypred_train, coeffs, intercepts = model.fit_predict(Train[xcols], Train[ycol], Train[xcols])
        else:

            model.fit(Train[xcols], Train[ycol])
            # if the fit is good, then predict the training set
            if model.goodfit:
                ypred_train = model.predict(Train[xcols])
            else:
                model = None
                modelkey = None

        # add the calibration predictions to the appropriate column
        if method == 'Local Regression':
            calcol = ('predict', '"' + method + '- Cal -' + str(paramgrid[i]) + ' n_neighbors: ' + str(
                n_neighbors) + '"')
        else:
            calcol = ('predict', '"' + method + '- Cal -' + str(paramgrid[i]) + '"')
        predictkeys.append(calcol[-1])
        Train[calcol] = ypred_train
        # append the RMSEC for the current settings to the cllection of all RMSECs
        output_tmp['RMSEC'] = RMSE(ypred_train, Train[ycol])

    output = output_tmp
    return output, model, modelkey, predictkeys, Train['predict']

class cv:
    def __init__(self, paramgrid):
        self.paramgrid = paramgrid

    def do_cv(self, Train, xcols='wvl', ycol=('comp', 'SiO2'), method='PLS',
              yrange=None, calc_path = False, alphas = None):

        if yrange is None:
            yrange = [np.min(Train[ycol]),np.max(Train[ycol])]

        args = list(range(len(self.paramgrid)))
        kwargs = {'paramgrid':self.paramgrid, 'Train':Train, 'xcols': xcols, 'ycol': ycol, 'method': method, 'yrange':yrange, 'calc_path':calc_path, 'alphas':alphas}
        t = time.time()
        results = Parallel(n_jobs= -1)(delayed(cv_core)(i, **kwargs) for i in args)
        t2 = time.time()
        parallel_t = t2-t

        models = []
        modelkeys = []
        predictkeys = []
        output = pd.DataFrame()
        for i in results:
            if i != 0:
                output = pd.concat((output, i[0]))

                if i[1] is not None:
                    models.append(i[1])
                if i[2] is not None:
                    modelkeys.append(i[2])
                if i[3] is not None:
                    for j in i[3]:
                        predictkeys.append(j)
                if i[4] is not None:
                    try:
                        cv_predicts = pd.merge(cv_predicts, i[4], left_index=True, right_index=True)
                    except:
                        cv_predicts = i[4]

        try:
            cv_predicts.columns = [('predict',i) for i in cv_predicts.columns]
            Train = pd.concat((Train,cv_predicts),axis=1)
        except:
            print('Unable to concatenate CV predictions with Training data')

        #make the columns of the output data drame multi-indexed
        try:
            cols = output.columns.values
            cols = [('cv', i) for i in cols]
            output.columns = pd.MultiIndex.from_tuples(cols)
        except:
            print('No valid output')


        return Train, output, models, modelkeys, predictkeys