import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import svm
from scipy import stats
import math
import random


def load_data(path_train='../input/train.csv',
              target='Horizontal_Distance_To_Fire_Points'):
    trainDf = pd.read_csv(path_train)
    testIDs = trainDf['ID']
    yTrain = trainDf[target]
    xTrain = trainDf.drop([target, 'ID'], 1)
    return xTrain, yTrain, testIDs


def transform_data(xTrain, yTrain):
    # Features
    CATEGORICAL_FEATURES = ['Soil_Type']
    NUMERIC_FEATURES = [feature for feature in list(xTrain.columns.values) if feature not in CATEGORICAL_FEATURES]

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    soil_classes = sorted(xTrain['Soil_Type'].unique())
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(categories=[soil_classes]))])  # handle_unknown='ignore'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)])

    fitProcessor = preprocessor.fit(xTrain)
    xTrain = fitProcessor.transform(xTrain)

    testsize = xTrain.shape[0]
    xTrain = xTrain[:testsize]
    yTrain = yTrain[:testsize]
    return xTrain,yTrain


def runCrossValidation(model,X,y, kfolds):
    scores = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds, n_jobs=-1))
    return scores.mean()

def run_multiple_CV(model, Xtrain, X_PCA, y, splits = 10, runs = 10):
    print("Testing: original data against PCA data with {} runs of {} folds".format(splits, runs))

    errors = []
    for i in range(0,runs):
        random_state = random.randint(0,12321343)
        kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        err1, err2 = runCrossValidation(model, Xtrain, y, kf), runCrossValidation(model, X_PCA, y, kf)
        errors.append((err1,err2))
    return np.array(errors)

def t_test(errors, threshold = .05):
    print("Errors: ")
    print(errors)
    a,b,n = errors[:,0],errors[:,1], errors.shape[0]
    # d = a-b
    #     # d_bar = d.mean()
    #     # sigma = math.sqrt(np.power((d-d_bar),2).sum() / n-1)
    #     # t = d_bar / (sigma * math.sqrt(n))
    t, a = stats.ttest_ind(a,b)
    print("Probability of observing result if means were equal: {}".format(a))
    is_significant = a < threshold
    res = ""
    if is_significant:
        if a.mean() > b.mean():
            res = "Result is Statistically Significant with alpha {}\n-->b performs better than a"
        else:
            res = "Result is Statistically Significant with alpha {}\n-->a performs better than b"
    else:
        res = "Result is not Statistically Significant with alpha {}\n"
    print(res.format(threshold))

def compare_data(model, Xtrain, X_PCA, y):
    errors =run_multiple_CV(model, Xtrain, X_PCA, y, splits=5, runs=5)
    t_test(errors)

