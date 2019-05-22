"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm
from sklearn.model_selection import KFold



def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, np.inf
    errors = np.zeros((len(paras), len(Cs)))

    # 10 fold cross validation
    splits = 10
    kf = KFold(n_splits=splits)

    for i, P in enumerate(paras):
        # print('P:',  P)
        for j, C in enumerate(Cs):
            # print('C:', C)
            averageTrainError = 0
            for train_index, test_index in kf.split(xTr.T):
                X_train, X_test = xTr[:, train_index], xTr[:, test_index]
                y_train, y_test = yTr[train_index], yTr[test_index]

                svmclassify = trainsvm(X_train, y_train, C, ktype, P)
                train_preds = svmclassify(X_test)
                averageTrainError = averageTrainError + np.mean(train_preds != y_test)

            averageTrainError = averageTrainError / float(splits)
            if averageTrainError < lowest_error:
                lowest_error = averageTrainError
                bestC = C
                bestP = P
                errors[i, j] = averageTrainError

    print(errors)

    return bestC, bestP, lowest_error, errors


    