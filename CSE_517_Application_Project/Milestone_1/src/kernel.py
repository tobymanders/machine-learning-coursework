#441702 rwelish@wustl.edu Welish Ryan
# X X tmanders@wustl.edu Manders Toby
#448882 sfoeller@wustl.edu Foeller Sebastian

import os
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score


#-----------------Loading Data------------------#

def loadTrainData(path='../input/train.csv'):
    '''
        input: path to file
        output:
        X: nxd
        y: 1xn
    '''
    df = pd.read_csv(path)
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")
    return df.drop(['Horizontal_Distance_To_Fire_Points','ID','Soil_Type'], axis=1), df['Horizontal_Distance_To_Fire_Points']

def loadTestData(path='../input/test.csv'):
    '''
        input: path to file
        output:
        X: nxd
        y: 1xn
    '''
    df = pd.read_csv(path)
    print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    return df.drop(['ID','Soil_Type'],axis=1),df["ID"]

# ---Attempt at feature engineering but only hurt performance---#
# def loadFeatureAddedTrainData(path='../input/train.csv'):
#     df = pd.read_csv(path)
#     inverse_cols =["dist_to_hydro"]
#     df["dist_to_hydro"] = np.sqrt(np.square(df["Horizontal_Distance_To_Hydrology"])+
#     np.square(df["Vertical_Distance_To_Hydrology"]))
#     df = df.drop(['Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology'],axis=1)
#     for col in inverse_cols:
#         df.loc[df[col] > 0, col] = 1/df[df[col]>0][col]
#     return df.drop(['Horizontal_Distance_To_Fire_Points','ID'], axis=1), df['Horizontal_Distance_To_Fire_Points']

#---------------------------------------------#

def rmse(yTr,yTe):
    '''input true labels and predictions - returns root mean square error'''
    return math.sqrt(mean_squared_error(yTr, yTe))

#-----------------Tuning and Cross Validation------------------#

def tuneAlpha(model):
    bestAlpha = -1
    bestScore = 10000
    for alpha in np.linspace(.0001,.001,num=20):
        regressor = model(alpha = alpha,normalize = True)
        score = runCrossValidation(regressor)
        if(score < bestScore):
            bestAlpha,bestScore = alpha,score
    print("Best alpha was {} with score {}".format(bestAlpha,bestScore))

def runCrossValidation(untrainedModel,folds=10):
    X,y = loadTrainData()
    scores = np.sqrt(-cross_val_score(untrainedModel, X, y, cv=folds,scoring="neg_mean_squared_error"))
    print("Error: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()
#---------------------------------------------#


#-----------------Load Data and Evaluate Predictions------------------#

def runExperiment(model, test_size=.2):
    '''
    Runs a model on the training data by splitting it into train/validation sets
    input:
        model: model to be run - function taking in (Xtr,yTr,Xte) returning predictions
    '''
    X,y = loadTrainData()
    xTr,xVal, yTr, yVal = train_test_split(X,y,test_size = test_size)
    preds = model(xTr,yTr,xVal)
    error = rmse(preds,yVal)
    print("Error of {} \n".format(error))
    return error

def predictOnTestData(model):
    '''
    Runs the model over test data and outputs prediction file
    input:
        model: model to be run - function taking in (Xtr,yTr,Xte) returning predictions
    '''
    print("Making predictions on test set")
    xTr,yTr = loadTrainData()
    xTe,IDs = loadTestData()
    preds =  model(xTr,yTr,xTe)
    my_submission = pd.DataFrame({'ID': IDs, 'Horizontal_Distance_To_Fire_Points': preds})
    my_submission.to_csv('submission.csv', index=False)


#-----------------Construct Models/ Make preds------------------#

def RidgeRegression(xTr,yTr,xTe,alpha = 0.05):
    '''
    Runs Ridge regression with given alpha - X is nxd and y is nx1
    '''
    ridgeRegressor = Ridge(alpha = alpha,normalize = True)
    ridgeRegressor.fit(xTr,yTr)
    return ridgeRegressor.predict(xTe)

def LassoRegression(xTr,yTr,xTe,alpha = 0.05):
    '''
    Runs Lasso regression with given alpha - X is nxd and y is nx1
    '''
    lassoRegressor = Lasso(alpha = alpha,normalize = True)
    lassoRegressor.fit(xTr,yTr)
    return lassoRegressor.predict(xTe)

def SVMRegression(xTr,yTr,xTe,alpha = 0.05):
    svmRegressor = LinearSVR(max_iter = 1000000)
    svmRegressor.fit(xTr,yTr)
    return svmRegressor.predict(xTe)


#-----------------Main------------------#

def main():
    #runExperiment(RidgeRegression)
    #runExperiment(LassoRegression)
    #runExperiment(SVMRegression)
    #predictOnTestData(RidgeRegression)
    tuneAlpha(Ridge)
    #tuneAlpha(Lasso)
    #runCrossValidation(LinearSVR(max_iter = 1000000),folds =5)

if __name__ == '__main__':
    main()
