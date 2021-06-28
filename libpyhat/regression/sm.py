# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:15:46 2016

@author: rbanderson
"""
import numpy as np
import scipy.optimize as opt
import copy

class sm:
    def __init__(self, blendranges, random_seed = None):
        self.blendranges = blendranges
        self.random_seed = random_seed

    def do_blend(self, predictions, truevals=None, verbose = True):
        # create the array indicating which models to blend for each blend range
        # For three models, this creates an array like: [[0,0],[0,1],[1,1],[1,2],[2,2]]
        # Which indicates that in the first range, just use model 0
        # In the second range, blend models 0 and 1
        # in the third range, use model 1
        # in the fourth range, blend models 1 and 2
        # in the fifth range, use model 2

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.toblend = []
        for i in range(len(predictions) - 1):
            self.toblend.append([i, i])
            if i < len(predictions) - 2:
                self.toblend.append([i, i + 1])

        # If the true compositions are provided, then optimize the ranges over which the results are blended to minimize the RMSEC
        blendranges = np.array(self.blendranges).flatten()  # squash the ranges to be a 1d array
        blendranges.sort()  # sort the entries. These will be used by submodels_blend to decide how to combine the predictions
        self.blendranges = blendranges
        if truevals is not None:
            self.rmse = 99999999
            n_opt = 5
            i=0
            while i < n_opt:
                if i > 0:
                    blendranges = np.hstack(([-9999], blendranges[1:-1]+0.1*np.random.random(len(blendranges)-2), [9999])) #add some randomness to the blendranges each time
                truevals = np.squeeze(np.array(truevals))
                result = opt.minimize(self.get_rmse, blendranges, (predictions, truevals), tol=0.00001)

                if result.fun < self.rmse:
                    self.blendranges = result.x
                    self.rmse = result.fun
                    if verbose==True:
                        print(self.blendranges.sort())
                        print('RMSE ='+str(self.rmse))
                else:
                    pass
                i=i+1
            print(' ')
            print('Optimum settings:')
            print('RMSE = ' + str(self.rmse))
            print('Low model: ' + str(round(self.blendranges[0], 4)) + ' to ' + str(round(self.blendranges[2], 4)))
            i = 1
            m = 2
            while i + 3 < len(self.blendranges) - 1:
                print('Submodel ' + str(m) + ': ' + str(round(self.blendranges[i], 4)) + ' to ' + str(
                    round(self.blendranges[i + 3], 4)))
                i = i + 2
                m = m + 1
            print('High model: ' + str(round(self.blendranges[-3], 4)) + ' to ' + str(round(self.blendranges[-1], 4)))

        else:
            self.blendranges = blendranges


       # print(self.blendranges)
        # calculate the blended results
        blended = self.submodels_blend(predictions, self.blendranges, overwrite=False)
        return blended

    def get_rmse(self, blendranges, predictions, truevals, rangemin = 0.0, rangemax = 100, roundval = 10):
        blendranges[1:-1][blendranges[1:-1] < rangemin] = rangemin  # ensure range boundaries don't drift below min
        blendranges[1:-1][blendranges[1:-1] > rangemax] = rangemax  # ensure range boundaries don't drift above max
        blendranges.sort()  # ensure range boundaries stay in order

        blended = self.submodels_blend(predictions, blendranges, overwrite=False)
        # calculate the RMSE. Round to specified precision as a way to control how long optimization runs
        # Note: don't want to round too much - optimization needs some wiggle room
        RMSE = np.round(np.sqrt(np.mean((blended - truevals) ** 2)),roundval)
        print('RMSE = '+str(RMSE))
        print('Low model: '+str(round(blendranges[0],4))+' to '+str(round(blendranges[2],4)))
        i=1
        m=2
        while i+3<len(blendranges)-1:
            print('Submodel '+str(m)+': '+str(round(blendranges[i],4))+' to '+str(round(blendranges[i+3],4)))
            i=i+2
            m=m+1
        print('High model: '+str(round(blendranges[-3],4)) + ' to ' + str(round(blendranges[-1],4)))

        return RMSE
        
    def submodels_blend(self,predictions,blendranges, overwrite=False):
        blended=np.squeeze(np.zeros_like(predictions[0]))
        
        #format the blending ranges (note, initial formatting is done in do_blend)
        blendranges = np.hstack((blendranges, blendranges[1:-1]))  # duplicate the middle entries
        blendranges.sort() #re-sort them
        blendranges=np.reshape(blendranges,(int(len(blendranges)/2),int(2)))  #turn the vector back into a 2d array (one pair of values for each submodel)
        self.toblend.append([len(predictions)-1,len(predictions)-1])
        blendranges=np.vstack((blendranges,[-9999999,999999]))
        #print(blendranges)
        for i in range(len(blendranges)): #loop over each composition range
            for j in range(len(predictions[0])): #loop over each spectrum
                ref_tmp=predictions[-1][j]   #get the reference model predicted value
                #check whether the prediction for the reference spectrum is within the current range
                inrangecheck=(ref_tmp>blendranges[i][0])&(ref_tmp<blendranges[i][1])
     
                if inrangecheck: 
                    try:
                        if self.toblend[i][0]==self.toblend[i][1]: #if the results being blended are identical, no blending necessary!
                            blendval=predictions[self.toblend[i][0]][j]

                        else:
                            weight1 = 1 - (ref_tmp - blendranges[i][0]) / (
                                blendranges[i][1] - blendranges[i][0])  # define the weight applied to the lower model
                            weight2 = (ref_tmp - blendranges[i][0]) / (
                                blendranges[i][1] - blendranges[i][0])  # define the weight applied to the higher model
                            # calculated the blended value (weighted sum)
                            blendval = weight1 * predictions[self.toblend[i][0]][j] + weight2 * \
                                                                                      predictions[self.toblend[i][1]][j]
                    except:
                        pass
                    if overwrite:
                        blended[j] = blendval  # If overwrite is true, write the blended result no matter what
                    else:
                        # If overwrite is false, only write the blended result if there is not already a result there
                        if blended[j] == 0:
                            blended[j] = blendval


        return blended


