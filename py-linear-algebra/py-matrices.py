'''
This file displayes Matrix Operations.
'''

import numpy as np

'Matrix A has 3 rows and 2 columns of 1s'
matrix_a = np.random.random((3, 2))

'Matrix B has 2 rows and 3 columns of 1s'
matrix_b = np.random.random((2, 3))

print("matrix a:\n{0}\nmatrix b: \n{1}".format(matrix_a, matrix_b))
print("Transpose of Matrix_a:\n", matrix_a.T)

'Dot Multiplication of matrix_a and matrix_b'
a_b_dot_product = np.dot(matrix_a, matrix_b)
print("\nDot product of Matrix A and matrix B is:\n", a_b_dot_product)

b_a_dot_product = np.dot(matrix_b, matrix_a)
print("\nDot product of Matrix B and Matrix A is:\n", b_a_dot_product)

matrix_a_sum_of_elements = matrix_a.sum()
print("\nSum of all memebers of matrix A is: ", matrix_a_sum_of_elements)

matrix_a_sum_of_column_elements = matrix_a.sum(axis = 0)
print("\nSum of elements at column level are:\n", matrix_a_sum_of_column_elements)

matrix_a_sum_of_row_elements = matrix_a.sum(axis = 1)
print("\nSum of elements at row level are: \n", matrix_a_sum_of_row_elements)

print('\nMatrix Minimum element:{0}\tMaximum element:{1}'.format(matrix_a.min(), matrix_a.max()))