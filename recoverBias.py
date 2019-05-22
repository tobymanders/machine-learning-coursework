"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

#T
def recoverBias(K,yTr,alphas,C):
    bias = 0

    # find max dist
    n, _ = yTr.shape
    temp = np.abs(alphas - C/2.0)
    ind = np.argmin(temp)
    k = K[ind, :].reshape(-1, 1)
    yi = yTr[ind]

    bias = yi - np.sum(alphas * yTr * k)


    return bias


#S
# def recoverBias(K,yTr,alphas,C):
#     bias = .5
#
#     # Get alphas closest to the midpoint between 0 and C
#     temp = np.abs(alphas - C / 2.0)
#     ind = np.argmin(temp)
#
#     y = yTr * alphas
#     kernel = K[ind:ind+1]
#     p = np.dot((yTr * alphas), kernel)
#     bias = yTr[ind] - np.dot(K[ind:ind+1], y)
#
#     print("Bias: ", bias)
#     return bias
#
