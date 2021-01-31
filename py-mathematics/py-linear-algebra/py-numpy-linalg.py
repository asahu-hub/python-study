import numpy as np
import matplotlib.pyplot as plt

'''
1. Numpy is fundamental package for scientific computing in Python.
2. The most important object of numpy is N-dimensional array object.
3. It can be used to perform Linear Algebra, Fourier Transformation and Random Number operations.
4. It is a building block for other mathematical packages (eg: scipy)
5. It is an Open Source library.
'''
print("Numpy Version: ", np.__version__)

a = np.random.random((3, 3))
b = np.random.random((3, 3))

print("\nVector a:\n", a, "\nVector b:\n", b)

a_trace = np.trace(a)
print("\nTrace of a is:\n", a_trace)

a_b_column_stack = np.column_stack((a, b))
print("\nColumn-stack of a and b is:\n", a_b_column_stack)

a_b_row_stack = np.row_stack((a, b))
print("\nRow stack of a and b is:\n", a_b_row_stack)

a_inverse = np.linalg.inv(a)
print("\nInverse of a is:\n", a_inverse)

a_points = np.asarray([np.arange(1, 3, 1), np.arange(2, 4, 1), np.arange(3, 5, 1), np.arange(4, 6, 1)])
b_points = np.arange(10, 22, 3)
print("\n", a_points, b_points)

m, c = np.linalg.lstsq(a_points, b_points, rcond=None)[0]
print("\nLeast Square of a with constant is:\n", m, c)

#_ = plt.plot(a_points, b_points, 'o', label='Original data', markersize=10)
#_ = plt.plot(a_points, (m * a_points) + c, 'r', label='Fitted line')
#_ = plt.legend()
#plt.show()

a_eigen_vectors = np.linalg.eig(a)
print("\nEigen Vectors of a are:\n", a_eigen_vectors)

a_eigen_values = np.linalg.eigvals(a)
print("\nEigen Values of a are:\n", a_eigen_values)
