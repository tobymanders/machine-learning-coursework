from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from load_data import load_data
from write_submission import write_submission
import numpy as np
#from matplotlib import pyplot as plt
from expand_features import expand_features

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# load data
[Xtr, Ytr, Xte, testID] = load_data()

# expand features
Xtr_expanded = expand_features(Xtr)
Xte_expanded = expand_features(Xte)

print('Xtr shape', Xtr_expanded.shape, 'Ytr shape', Ytr.shape, 'Xte shape', Xte.shape)

clf = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1], normalize=True, store_cv_values=True).fit(Xtr_expanded, Ytr)
ridge_preds = clf.predict(Xtr_expanded)
print('RMSE Ridge:', rmse(ridge_preds, Ytr))
print(clf.alpha_)

# clf = kernel_ridge.KernelRidge(alpha=0.01)
# clf.fit(Xtr_expanded, Ytr)
# kridge_preds = clf.predict(Xtr_expanded)
# print('RMSE kridge:', rmse(kridge_preds, Ytr))
#
# clf = linear_model.Lasso()
# clf.fit(Xtr_expanded, Ytr)
# lasso_preds = clf.predict(Xtr_expanded)
# print('RMSE Lasso:', rmse(lasso_preds, Ytr))
#
# clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
# clf.fit(Xtr, Ytr)
# svm_preds = clf.predict(Xtr)
# print('RMSE SVM:', rmse(svm_preds, Ytr))
#
# plt.plot(svm_preds)
#
preds = clf.predict(Xte_expanded)
write_submission(testID, preds)
