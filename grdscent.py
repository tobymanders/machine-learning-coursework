
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
    old_loss = np.inf
    w = w0
    w_new = w
    c = float(stepsize)
    i = 0

    loss_hist = [np.inf]
    while i < maxiter:

        loss_old, gradient_old = func(w)
        loss_new, gradient_new = func(w_new)
        if np.linalg.norm(gradient_new) < tolerance:
            break

        # Check the last update. If the loss increased, revert w and update c.
        elif loss_new > loss_old:
            # print('Reset')
            c *= 0.5
            w_new = w
            i -= 1
        # Save w, loss. Prepare update.
        else:
            w = w_new
            # print('Success')
            # loss_hist.append(loss)
            c *= 1.01
            update = np.dot(gradient_new, c)
            w_new = w - update
            i += 1
    return w
