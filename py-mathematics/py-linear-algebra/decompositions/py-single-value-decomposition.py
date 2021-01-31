'''
Single Value Decomposition (SVD) can be used to decompose any matrix (irrespective of number of rows and columns) in its constituent parts to make certain subsequent matrix calculations simpler. Every matrix has a SVD.
'''

import numpy as np
from numpy import diag, zeros
from scipy.linalg import svd

matrix_a = np.array([[0, 3, 6, 10], [10,13,16,20],[20,23,26,30],[30,33,36,40]])
matrix_b = np.array([[0, 3], [10,13],[20,23]])
print("\nMatrix A:\n", matrix_a, "\nMatrix B:\n", matrix_b)

''' 
SVD Decomposition - 
A = UEV.T
    A = Any matrix (nxm)
    U = Matrix (mxm). It contains the left singular vectors of A.
    E = Diagonal matrix (mxn). It contains the singular values of A.
    V.T = Transpose of nxn matrix. V contains the right singular vectors of A.
'''

U, E, V = svd(matrix_a)
print("\nU:\n{0}\nE:\n{1}\nV:\n{1}\n".format(U, E, V))

# Reconstructing Original Matrix

# Case-1: When the parent matrix is a square matrix. (nxn)
sigma = diag(E)
svd_square_matrix_reconstruction = U.dot(sigma.dot(V))
print("\nReconstructed Square Matrix using SVD:\n", svd_square_matrix_reconstruction)


# Case-2: When parent matrix is not a square matrix. (nxm)
U_1, E_1, V_1 = svd(matrix_b)
sigma_1 = zeros((matrix_b.shape[0], matrix_b.shape[1]))
sigma_1[: matrix_b.shape[1], :matrix_b.shape[1]] = diag(E_1)
svd_rectangular_matrix_reconstruction = U_1.dot(sigma_1.dot(V_1))
print("\nReconstructed Rectangular Matrix using SVD:\n", svd_rectangular_matrix_reconstruction)
