from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regression constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w

    ywx = np.multiply(yTr, np.matmul(w.T, xTr))
    diff = np.subtract(1, ywx)

    loss = np.sum(diff[ywx < 1]) + lambdaa * np.dot(w.T, w)

    diff_neg = np.broadcast_to(diff < 0, xTr.shape)

    x_mod = xTr
    x_mod[diff_neg] = 0

    gradient = np.matmul(x_mod, -yTr.T) + 2 * lambdaa * w

    return loss, gradient
