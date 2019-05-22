import numpy as np

def linearmodel(w,xTe):
# INPUT:
# w weight vector (default w=0)
# xTe dxn matrix (each column is an input vector)
#
# OUTPUTS:
#
# preds predictions

    preds = np.dot(w.T, xTe)
    # preds = []
    # for i in range(xTe.shape[1]):
    #     prediction = np.dot(w.T, xTe[:, i])
    #     preds.append(prediction)

    return preds
