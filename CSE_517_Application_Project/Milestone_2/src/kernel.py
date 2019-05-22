#441702 rwelish@wustl.edu Welish Ryan
# X X tmanders@wustl.edu Manders Toby
#448882 sfoeller@wustl.edu Foeller Sebastian

import os
import pyGPs
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#-----------------Loading Data------------------#

def loadTrainData(path='../input/train.csv', nrows = 14302):
    '''
        input: path to file
        output:
        X: nxd
        y: 1xn
    '''
    df = pd.read_csv(path, nrows = nrows)
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    # df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    # df["Soil_Type_4"] = df["Soil_Type"] % 10
    #df[df['From_Cache_la_Poudre'] == 0]['From_Cache_la_Poudre'] = -1
    df.loc[df['From_Cache_la_Poudre'] == 0,'From_Cache_la_Poudre'] = -1
    print("Loaded {}. Shape: {}".format(os.path.basename(path),df.shape))
    return df.drop(['From_Cache_la_Poudre','ID','Soil_Type'], axis=1), df['From_Cache_la_Poudre'].values

def loadTestData(path='../input/test.csv'):
    '''
        input: path to file
        output:
        X: nxd
        y: 1xn
    '''
    df = pd.read_csv(path)
    print("Loaded {}. Shape: {}".format(os.path.basename(path),df.shape))
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    # df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    # df["Soil_Type_4"] = df["Soil_Type"] % 10
    return df.drop(['ID','Soil_Type'],axis=1),df["ID"]



#---------------------------------------------#
def predictOnTestData(nrows = 14302):
    '''
    Runs the model over test data and outputs prediction file
    input:
        model: model to be run - function taking in (Xtr,yTr,Xte) returning predictions
    '''
    print("Making predictions on test set")
    xTr,yTr = loadTrainData(nrows = nrows)
    xTe,IDs = loadTestData()
    preds, probs =  getSK_preds(xTr,yTr,xTe)
    print(probs)
    my_submission = pd.DataFrame({'ID': IDs, 'From_Cache_la_Poudre': preds})
    my_submission.to_csv('submission.csv', index=False)

def getSK_preds(xTr,yTr,xTe,a=(54.3 ** 2),l=(2.23e+03)):
    kernel = a * RBF(l)
    gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer = 1).fit(xTr, yTr)
    return gpc.predict(xTe), gpc.predict_proba(xTe)

## ------None of this works at the moment-----#####
# --------------------------------------------#####
# def getGP_preds(xTr,yTr,xTe):
#     model = pyGPs.gp.GPC()  # binary classification (default inference method: EP)
#     model.getPosterior(xTr, yTr)  # fit default model (mean zero & rbf kernel) with data
#     model.optimize(xTr, yTr)  # optimize hyperparamters (default optimizer: single run minimize)
#     return model.predict(xTe)
#
# def crossValidate(xTr,yTr,model):
#     scores = cross_val_score(model, xTr, yTr, cv=5, scoring = 'roc_auc')
#     return scores.mean(), scores.std()
#
# def runCrossValidation(kernels, nrows = 14302):
#     xTr,yTr = loadTrainData(nrows = nrows)
#     results = []
#     best_kernel, best_mean = '',100000
#     for kernel in kernels:
#             gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=1).fit(xTr, yTr)
#             mean,std = crossValidate(xTr,yTr,gpc)
#             results.append((mean,std))
#             if mean < best_mean:
#                 best_kernel,best_mean = gpc.kernel_,mean
#     print(results)
#     print(best_kernel)
#     print(best_mean)
# --------------------------------------------#####
## ------None of this works at the moment-----#####


#-----------------Main------------------#

def main():
    predictOnTestData()



if __name__ == '__main__':
    main()
