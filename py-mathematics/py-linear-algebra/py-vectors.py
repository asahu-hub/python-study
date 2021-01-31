'''
    The code below dispalys how to perform vector operations using python numpy library.
'''

import numpy as np
print(np.__version__)


a=np.arange(2, 10, 3)
b=np.arange(3, 11, 3)

print("vector a: ", a, "\nvector b: ", b)

a_b_outer_product = np.outer(a, b)
print("\nOuter product of a and b is: \n", a_b_outer_product)
print("\nShape of outer product is: ", a_b_outer_product.shape)

a_b_inner_product = np.inner(a, b)
print("\nInner product of a and b is: \n", a_b_inner_product)

' Dot product is same as inner product'
a_b_dot_product = np.dot(a, b)
print("\nDot Product of a and b is:\n", a_b_dot_product)

a_b_cross_product = np.cross(a, b)
print("Cross product of a and b is:\n", a_b_cross_product)