import numpy as np
from numpy.core.numeric import ones
'''
1. Numpy is fundamental package for scientific computing in Python.
2. The most important object of numpy is N-dimensional array object.
3. It can be used to perform Linear Algebra, Fourier Transformation and Random Number operations.
4. It is a building block for other mathematical packages (eg: scipy)
5. It is an Open Source library.
'''
print("Numpy Version: ", np.__version__)

a = [1,-25,33,12]
b = [51,26,-7,38]

onedArray = np.array(a)
twodArray = np.array([a, b])
print("1-D Array: \n", onedArray, "\n\n2-D Array: \n", twodArray)
print("\n2-D Array Dimension: ", twodArray.ndim, "\n2-D Array Shape (MxN): ", twodArray.shape, "\n2-D Number of Elements: ", 
twodArray.size, "\n2-D Data Type: ", twodArray.dtype) 

oneDArrayTranspose = onedArray.T
twodArrayTranspose = twodArray.T
print("\n1-D Array Transpose: \n", oneDArrayTranspose, "\n2-D Array Transpose: \n", twodArrayTranspose)
print("\n2-D Array T Dimension: ", twodArrayTranspose.ndim, "\n2-D T Array Shape (MxN): ", twodArrayTranspose.shape, "\n2-D T Number of Elements: ", 
twodArrayTranspose.size, "\n2-D T Data Type: ", twodArrayTranspose.dtype)

# Generating Array using patterns
oddNumbersArray = np.arange(1, 10, 3, int)
print("\nOdd Numbers array: \n", oddNumbersArray, " with data type: ", oddNumbersArray.dtype)

# Creating a matrix with 2 rows and 3 columns, with all zeroes
zerosArray = np.zeros((2 ,3))
print("\nZeroes array: \n", zerosArray)

# Creating a matrix with 3 rows and 4 columns, with all ones
onesArray = np.ones((3, 4))
print("\nOnes Array: \n", onesArray)

# Creating a diagonal matrix with 4 rows and 5 columns.
x = np.array([np.arange(1, 4, 1), np.arange(2, 5, 1), np.arange(3, 6, 1)])
print("\nOriginal Array: \n", x)
diagArray = np.diag(x)
print("\nDiagonal Array: \n", diagArray)