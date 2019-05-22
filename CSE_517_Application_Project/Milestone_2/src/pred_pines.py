#!C:\Users\Toby-PC\Desktop\milestone2\Scripts\python.exe
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

def loadTrainData(path='../data/train.csv'):
    df = pd.read_csv(path)
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    # print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")
    return df.drop(['From_Cache_la_Poudre','ID', 'Soil_Type'], axis=1), df['From_Cache_la_Poudre']

def loadTestData(path='../data/test.csv'):
    df = pd.read_csv(path)
    df["Soil_Type_1"] = df["Soil_Type"] // 1000
    df["Soil_Type_2"] = df["Soil_Type"] % 1000 // 100
    df["Soil_Type_3"] = df["Soil_Type"] % 100 // 10
    df["Soil_Type_4"] = df["Soil_Type"] % 10
    # print(f"Loaded {os.path.basename(path)}. Shape: {df.shape}")
    return df.drop(['ID', 'Soil_Type'],axis=1), df["ID"]

def l2distance(x, y):
    m = x.shape[0] # x has shape (m, d)
    n = y.shape[0] # y has shape (n, d)
    x2 = np.sum(x**2, axis=1).reshape((m, 1))
    y2 = np.sum(y**2, axis=1).reshape((1, n))
    xy = x.dot(y.T) # shape is (m, n)
    dists = x2 + y2 - 2*xy # shape is (m, n)
    return dists

def write_submission(testID, preds):
    f = 'submission.csv'
    with open(f, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        row = ['ID', 'From_Cache_la_Poudre']
        csvwriter.writerow(row)
        for ID, pred in zip(testID, preds):
            csvwriter.writerow([ID, pred])

# Timer
start = time.time()

# Load data
X, y = loadTrainData()
soils = X[['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4']].copy()
soils = pd.get_dummies(soils.astype(str))
print(f'X columns: {soils.columns}')
X = X.drop(['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4'], axis=1)
X, y = X.values, y.values
X = np.concatenate((X, soils), axis=1)

# Data preprocessing
# num_to_use = 9000
# X, y = X[:num_to_use, :], y[:num_to_use]
y = y.reshape(-1, 1)

# Basic feature engineering
X[:, 1] = np.sin(X[:, 1] * np.pi / 180)

# Data scaling
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
means[10:] = 0
stds[10:] = 1
X = (X - means) / stds

# Split data for crossvalidation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train = X, y

# Covariance parameters
gammas = [2.1] # Chosen from crossvalidation
alphas = [0.9]
N = len(y_train)

auc_log = np.zeros((len(gammas), len(alphas)))
for gind, gamma in enumerate(gammas):
    for aind, alpha in enumerate(alphas):
        # Compute covariance for training data.
        C = alpha * np.exp(-gamma * l2distance(X_train, X_train))

        # Invert C now for performance.
        f = np.zeros((N, 1))
        invC = np.linalg.inv(C) # This takes a while.

        # Newton-Raphson procedure for finding MAP f. (This takes a really long time.)
        max_its = 10
        allf = np.zeros((max_its, N))
        allf[0, :] = f.T
        for it in range(max_its - 1):
            g = np.divide(1, (1 + np.exp(-f)))
            gradient = np.dot(-invC, f) + y_train - g
            hessian = -invC - np.diagflat(g * (1-g))
            f -= np.dot(np.linalg.pinv(hessian), gradient)
            allf[it+1, :] = f.T
        hessian = -invC - np.diag(g * (1-g))


        # # Evaluation
        # # Covariance functions
        # R = alpha * np.exp(-gamma * l2distance(X_train, X_test)**2)
        # # Cstar = alpha * np.exp(-gamma * l2distance(testx, testx)**2)
        #
        # # Latent function
        # fs = np.dot(np.dot(R.T, invC), f)
        #
        # # Probabilistic predictions
        # predprobs = np.concatenate(1 / (1 + np.exp(-fs)), axis=0)
        # # preds = predprobs > 0.5
        #
        # # Accuracy score
        # fpr, tpr, thresholds = roc_curve(y_test, predprobs)
        # auc_log[gind, aind] = auc(fpr, tpr)

# Report best gamma and alpha
a, b = np.unravel_index(auc_log.argmax(), auc_log.shape)
print(f'Best gamma: {gammas[a]}, Best alpha:{alphas[b]}')
print('AUC:', auc_log[a, b])
print(f'all AUC {auc}')

print(f'Time elapsed: {time.time() - start}')

# Load test set.
Xte, IDs = loadTestData()
soils = Xte[['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4']].copy()
soils = pd.get_dummies(soils.astype(str))
print(f'Xte columns: {soils.columns}')
Xte = Xte.drop(['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4'], axis=1)
Xte = Xte.values
Xte = np.concatenate((Xte, soils), axis=1)


# Basic feature engineering
Xte[:, 1] = np.sin(Xte[:, 1] * np.pi / 180)

# Use stored means and stds for data scaling.
Xte = (Xte - means) / stds


# Covariance functions
R = alpha * np.exp(-gamma * l2distance(X_train, Xte)**2)

# Latent function
fs = np.dot(np.dot(R.T, invC), f)

# Probabilistic predictions
preds = 1 / (1 + np.exp(-fs))
preds = np.concatenate(preds, axis=0)

# Output
write_submission(IDs, preds)