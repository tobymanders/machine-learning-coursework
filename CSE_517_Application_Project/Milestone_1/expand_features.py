import numpy as np

def expand_features(X):
    X = np.array(X)

    # euclidean distance
    euclidean_dist = np.mat(np.sqrt(X[:, 3]**2 + X[:, 4]**2)).T
    print(euclidean_dist.shape)
    # # soil digits
    soil = X[:, 9]
    # # print(soil)
    digits = np.divide(soil, 100).astype(int)
    # print(digits.shape)
    dig_1 = np.mat(np.divide(digits, 10).astype(int)).T
    dig_2 = np.mat(digits % 10).T
    # print(dig_1.shape)
    # print(dig_2.shape)
    # #
    # print(X.shape)
    Xnew = np.concatenate((X, dig_1, dig_2), axis=1)
    # print(Xnew.shape)
    return Xnew
