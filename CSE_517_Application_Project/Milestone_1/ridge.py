from linearmodel import linearmodel
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # preds = np.matmul(xTr.T, w)
    # difference = preds.T - yTr
    #
    # squared_difference = float(np.dot(difference, difference.T))
    # regularizer = float(lambdaa * np.dot(w.T, w))
    #
    # loss = squared_difference + regularizer
    #
    # reg = 2 * lambdaa * w
    # diff_x = 2 * np.matmul(xTr, difference.T)
    # gradient = diff_x + reg
    X = xTr.T
    Y = yTr.T
    t = (np.dot(X, w) - Y)

    loss = np.dot(t.T, t) + lambdaa * np.dot(w.T, w)
    gradient = 2 * np.dot(xTr, t) + 2 * lambdaa * w

    # Summation Notation Version.
    # num_samples = xTr.shape[1]
    # dims = w.shape[0]
    # w_norm2 = np.dot(w.T, w)
    # loss_sum = 0
    # grad_sum = np.zeros((dims, 1))
    # for i in range(num_samples):
    #     x_i = xTr[:, i]
    #     x_i.shape = w.shape
    #     wx_minus_y = float(np.dot(w.T, xTr[:, i]) - yTr[0, i])
    #     loss_sum += wx_minus_y ** 2
    #     grad_sum = np.add(wx_minus_y * x_i, grad_sum)
    # loss = loss_sum + lambdaa * w_norm2
    # gradient = np.add(2 * grad_sum, 2 * lambdaa * w)



    
    return loss, gradient
