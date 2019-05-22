"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""
import numpy as np
from cvxopt import matrix

# def generateQP(K, yTr, C):
#     yTr = yTr.astype(np.double)
#     n = yTr.shape[0]
#
#     Q = yTr.dot(yTr.T) * K
#     p = -np.ones((n, 1))
#     A = yTr.T
#     b = np.zeros((1, 1))
#
#     h = C * np.ones((n, 1))
#     G = np.identity(n)
#
#
#     # Q = matrix(Q)
#     # p = matrix(p)
#     # G = matrix(G)
#     # h = matrix(h)
#     # A = matrix(A)
#     # b = matrix(b)
#     return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)
#     # return Q, p, G, h, A, b


#S
def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]

    Q = np.array(np.dot(yTr, yTr.T)*K)
    p = np.array(-1*np.ones([n, 1]))
    b = np.zeros([1, 1])
    A = np.array(yTr.T)

    # Alpha <= C
    h1 = C * np.ones([n, 1])
    # Alpha >= 0
    h2 = np.zeros([n, 1])
    h = np.concatenate((h1, h2))

    # Alpha <= C
    g1 = np.identity(n)
    # Alpha >= 0
    g2 = -1*np.identity(n)
    G = np.concatenate((g1, g2))

    return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

