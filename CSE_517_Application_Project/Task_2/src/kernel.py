import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import warnings


# --- Load Data --- #
def loadTrainData(path='../input/train.csv', nrows=14302):
    """
    :param path: path to file
    :param nrows: rows of training data to read
    :return: X: nxd,y: 1xn
    """
    df = pd.read_csv(path, nrows=nrows)
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    print("\tLoaded {}. Shape: {}".format(os.path.basename(path), df.shape))
    return df.drop(['From_Cache_la_Poudre', 'ID', 'Soil_Type'], axis=1), df['From_Cache_la_Poudre'].values


def loadTestData(path='../input/test.csv'):
    """
    load test data into dataframe
    :param path: path of file
    :return: testData nxd, IDs nx1
    """
    df = pd.read_csv(path)
    print("\tLoaded {}. Shape: {}".format(os.path.basename(path), df.shape))
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    return df.drop(['ID', 'Soil_Type'], axis=1), df["ID"]


# --- normalization and feature Engineering --- #
def feature_engineer(df):
    """
    Adds features to training dataset. currently - One hot encoding of soil, sin of Aspect, cos of Aspect
    :param df: dataframe to add features to
    :return: new dataframe with added features
    """
    # df.assign(sin_Aspect=np.sin(df['Aspect']))
    df.Aspect = np.sin(np.deg2rad(df.Aspect))
    #df.assign(cos_Aspect=np.cos(df['Aspect']))
    return pd.get_dummies(df, columns=['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_2', 'Soil_Type_2'])


def normalize(xTr, xTe):
    """
    Normalizes both xTr and xTe.
    :param xTr: train data
    :param xTe:  test data
    :return: train data and test data normalized wrt train data
    """
    scaler = StandardScaler()
    scaler.fit(xTr)
    return scaler.transform(xTr), scaler.transform(xTe)


# --- Cross Validation and Tuning Parameters --- #
def run_cross_validation(untrainedModel, folds=10, featureEngineering=False):
    """
    Runs Cross validation on the given model and all training data split into folds
    :param untrainedModel: model to use for cross validation
    :param folds: number of folds for cross validation
    :param featureEngineering: should feature engineering be used
    :return: mean accuracy score over all folds
    """
    print("Running {}-Fold Cross Validation\n Feature Engineering: {}".format(folds, featureEngineering))
    X, y = loadTrainData()
    X = feature_engineer(X) if featureEngineering else X
    X = StandardScaler().fit_transform(X)
    scores = cross_val_score(untrainedModel, X, y, cv=folds, scoring="roc_auc", n_jobs=-1)
    print("\tAUC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def tune_gamma(gammas=[.01, 1], folds=5, featureEngineering=False):
    """
    Uses Cross Validation to find best performing gamma for smv.SVC with RBF kernel
    :param gammas: set of gammas to test
    :param folds: folds for cross validation
    :param featureEngineering: should feature engineering be used
    :return: best gamma value
    """
    accs = np.ones(len(gammas))
    for g_i, gamma in enumerate(gammas):
        model = svm.SVC(gamma=gamma)
        mean_acc = run_cross_validation(model, folds, featureEngineering)
        accs[g_i] = mean_acc
    best = accs.argmax()
    print("Best Gamma is {} with auc {}".format(gammas[best], accs[best]))
    return gammas[best]


def predict_on_test_data(model, gamma=1.7, featureEngineering=True):
    """
    Make predictions on the test dataset using entire training dataset. Writes results to submission.csv
    :param model: model to use to make predictions (function)
    :param gamma: gamma to use for RBF kernel
    :param featureEngineering: should model do feature engineering
    """
    print("Making predictions on test set")
    xTr, yTr = loadTrainData()
    xTe, IDs = loadTestData()
    preds = model(xTr, yTr, xTe, gamma=gamma, featureEngineering=featureEngineering)
    my_submission = pd.DataFrame({'ID': IDs, 'From_Cache_la_Poudre': preds[:, 1]})
    my_submission.to_csv('submission.csv', index=False)


# --- MODEL --- #
def svm_classify(xTr, yTr, xTe, featureEngineering=True, kernel='rbf', gamma=1.7):
    """
    Runs SVM Classification
    :param xTr: Training data
    :param yTr: training labels
    :param xTe: Test Data to classify
    :param featureEngineering: should feature engineering be performed
    :param kernel: kernel to use in SVM
    :param gamma: gamma parameter to use for RBF kernel
    :return: predicted labels for xTe
    """
    xTr, xTe = (feature_engineer(xTr), feature_engineer(xTe)) if featureEngineering else (xTr, xTe)
    xTr, xTe = normalize(xTr, xTe)
    svmClass = svm.SVC(gamma=gamma, kernel=kernel, probability=True)
    svmClass.fit(xTr, yTr)
    return svmClass.predict_proba(xTe)

def exploreFeature(X,y, feature):
    print(X[feature].describe())
    plt.scatter(X[feature],y)
    plt.show()

def eda(X,y):
    """
    Exploring the features. Some useful charts in here but unsure how to make best use of them
    :param X:
    :param y:
    """
    categorical = ['Soil_Type_1','Soil_Type_2']
    continuous = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']
    X[continuous].boxplot()
    plt.show()
    for cat in continuous:
        X.loc[y==0,cat].plot.kde(legend=True, label="Y=0")
        X.loc[y==1,cat].plot.kde(legend = True, label="Y=1")
        plt.title("{} distribution".format(cat))
        plt.show()

def our_kernel(X, Z):
    return Matern.__call__(X, Z)

# MAIN #
def main():
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    # run_cross_validation(svm.SVC(gamma = 'scale',kernel='poly'),folds = 5)
    # run_cross_validation(svm.SVC(gamma = 'scale'),folds = 5, featureEngineering=True)
    best = tune_gamma(gammas=[1.4], folds=4, featureEngineering=True)
    #tune_gamma(gammas=[1.6, 1.7], folds=4, featureEngineering=False)
    predict_on_test_data(svm_classify, gamma=best, featureEngineering=True)
    #xTr, yTr = loadTrainData()
    #eda(xTr,yTr)

if __name__ == '__main__':
    main()
