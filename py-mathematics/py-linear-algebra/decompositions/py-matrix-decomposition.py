'''
Matrix Decompositions are methods that reduce a matrix into constituent parts that make it easirer to calculate more complex matrix operations.

Matrix Decomposition is also called as Matrix Factorization.
'''

import numpy as np
import scipy as sp
from scipy.linalg import lu, qr, cholesky

print("\nNumpy Version:\t", np.__version__, "\nScipy Version:\t", sp.__version__)

matrix_a = np.array([[0, 3, 6, 10], [10,13,16,20],[20,23,26,30],[30,33,36,40]])
print("\nMatrix A: ", matrix_a)

''' 
LU Decomposition - It is for square matrices that are to be decomposed into Lower and Upper triangle matrices.
A = LU
    A = Square Matrix
    L = Lower Triangle Matrix
    U = Upper Triangle Matrix

Another variation is A = LUP
    P = Partial Pivoting matrix that specifices a way to permute the result or return the result to the original order.
'''
P, L, U = lu(matrix_a)
print("\nLower Triangular Matrix (L):\n{0}\nUpper Triangular Matrix:\n{1}\nPartial Pivoting Matrix:\n{1}".format(L, U, P))

'''
Reconstuct original matrix from P, L, U
'''
LU_reconstructed_matrix_a = P.dot(L).dot(U)
print("\nLU Reconstructed matrix is:\n", LU_reconstructed_matrix_a)


''' 
QR Decomposition - It is not only for square matrices, but for any nxm matrices.
A = QR
    A = Any Matrix of size (mxn)
    Q = Square matrix of size (mxm)
    R = Upper Triangle Matrix of size (mxn)

QR Decomposition can fail for matrices that cannot be decomposed or can be decomposed easily.
'''
Q, R = qr(matrix_a)
print("\nMatrix Q:\n{0}\nMatrix R:{1}".format(Q, R))

QR_reconstructed_matrix_a = Q.dot(R)
print("\nQR Reconstructed Matrix:\n", QR_reconstructed_matrix_a)


''' 
Cholesky Decomposition - It is only for square symmetric matrices, where all the elements of a matrix are non-zero (Positive Definite Matrix).
A = LL.T (or) UU.T
    A = Square Symmetric Matrix
    L = Lower Triangular Matrix
    L.T = Transpose of L
    U = Upper Triangular Matrix
    U.T = Transpose of U
'''
matrix_b = np.array([[2,1,1],[1,2,1],[1,1,2]])
print("\nMatrix B:\n", matrix_b)

L = cholesky(matrix_b)
print("\nCholesky Decomposed Lower Triangular Matrix:\n{0}".format(L))

Cholesky_reconstructed_matrix_L = L.dot(L.T)
print("\nCholesky Reconstructed Matrix:\n", Cholesky_reconstructed_matrix_L)
