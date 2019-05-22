
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):

    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Feel free to change this code any way you want
    
    f = lambda w : ridge(w,xTr,yTr,.1)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-16,10000)
    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained
