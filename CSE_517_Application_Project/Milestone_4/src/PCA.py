import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from sklearn import decomposition
from cross_validation import *
from sklearn import svm



def get_pca(xTrain):

    pca = decomposition.PCA(n_components=3)
    pca.fit(xTrain)
    X = pca.transform(xTrain)

    return X

if __name__ == "__main__":
    xTrain, yTrain, testIDs = load_data()
    xTrain, yTrain = transform_data(xTrain, yTrain)
    X_PCA = get_pca(xTrain)
    model = svm.SVR(kernel="rbf", C=55.0, gamma=.475)
    compare_data(model, xTrain, X_PCA, yTrain)