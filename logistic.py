import math
import numpy as np

import timeit

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    ywx = np.multiply(yTr, np.dot(w.T, xTr))
    e_ywx = np.exp(ywx)
    e_ywx_plus_one = np.add(e_ywx, 1)
    loss = np.sum(np.log(e_ywx_plus_one / e_ywx))

    numerator = np.multiply(yTr, xTr)
    div = numerator / e_ywx_plus_one

    gradient = -np.sum(div, axis=1)
    gradient.shape = w.shape

    return loss, gradient
