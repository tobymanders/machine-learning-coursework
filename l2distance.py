import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):

    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'

    D = np.zeros((n, m))
    x = X.T
    y = Z.T
    m = x.shape[0] # x has shape (m, d)
    n = y.shape[0] # y has shape (n, d)
    x2 = np.sum(x**2, axis=1).reshape((m, 1))
    y2 = np.sum(y**2, axis=1).reshape((1, n))
    xy = x.dot(y.T) # shape is (m, n)
    dists = x2 + y2 - 2*xy # shape is (m, n)
    return dists

    return D

#S
# def l2distance(X, Z):
#     d, n = X.shape
#     dd, m = Z.shape
#     assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
#
#     D = np.zeros((n, m))
#     X = X.T
#     for i in range(n):
#         D[i] = ((Z - X[i:i + 1].T) ** 2).sum(0)
#
#     return D
