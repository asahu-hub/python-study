import numpy as np
from numpy.linalg import inv, eig
from numpy import diag, array
import scipy as sp


print("\nNumpy Version:{0}\nScipy Version:{1}".format(np.__version__, sp.__version__))

matrix_a = np.array([[0, 3, 6, 10], [10,13,16,20],[20,23,26,30],[30,33,36,40]])
print("\nMatrix A:\n", matrix_a)

'''
Eigen Decomposition of a matrix is a type of decomposition that involves decomposing a square matrix into a set of eigent vectors and eigen values.

Eigen is a german word meaning Owner or Innate (Natural)
Eigen Vectors are unit vectors (length and magnitude = 1)
Eigen Values are scalars that act as co-efficients of the eigen vectors.

Av = Lv
    A = Square Matrix that is to be decomposed/factorized.
    v = Eigen Vector of A.
    L(Lambda) = Eigen Values of A.

Reconstructing A: A = QLQ.T
    Q = Matrix holding eigen vectors of A
    Q.T = Transpose of Q
    L = Diagonal Matrix of the eigen values.

A vector 'x' is an eigen vector of A, if Ax = Lx (i,e ...) If we multiply A with x, then you can get the same result by multiplying scalar L with 'x'.
'''
eigen_values, eigen_vectors = eig(matrix_a)
print("\nEigen Values:\n{0}\nEigen Vectors:\n{1}".format(eigen_values, eigen_vectors))

# Confirming Ax = Lx
'''
Note:
    1. Eigen Vectors are same dimension matrix as parent matrix. Every column of Eigen Vectors is a Eigen Vector.
    2. Eigen Values are always a 1-D list of values, where each index is a eigen value for that column.
        (i,e ...) First Element of Eigen Values is an Eigen Value for 1 Column of Eigen Vectors.
'''
first_column_eigen_vector = eigen_vectors[:, 0]
first_value_eigen_value = eigen_values[0]

Ax = matrix_a.dot(first_column_eigen_vector)
Lx = first_column_eigen_vector * first_value_eigen_value
print("\nAx:\n{0}\nLx:\n{1}".format(Ax, Lx))

# Reconstructing original array from eigen values and eigen vectors
Q = eigen_vectors
L = diag(eigen_values)
QT = inv(eigen_vectors)

print("\nQ:\n{0}\nL:\n{1}\n".format(QT, L))

reconstructed_matrix = Q.dot(L).dot(QT)
print("\nReconstructed Matrix:\n", reconstructed_matrix)

