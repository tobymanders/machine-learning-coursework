import csv
import numpy as np


def load_data():
    # Import data.
    fname = 'input/train.csv'
    Xtr_t = []
    Ytr_t = []
    # training data
    with open(fname) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            Xtr_ti = []
            for item in row[1:-1]:
                Xtr_ti.append(int(item))
            Ytr_t.append(int(row[-1]))
            Xtr_t.append(Xtr_ti)
        Xtr = np.array(Xtr_t)
        Ytr = np.array(Ytr_t)


    # print(Xtr.shape)
    # print(Ytr.shape)


    # testing data
    fname = 'input/test.csv'
    Xte_t = []
    testID = []
    with open(fname) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            Xte_ti = []
            for item in row[1:]:
                Xte_ti.append(int(item))
            testID.append(int(row[0]))
            Xte_t.append(Xte_ti)
        Xte = np.array(Xte_t)

    # Cross validation
    # create k folds of data.
    return Xtr, Ytr, Xte, testID




