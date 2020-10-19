
# coding: utf-8

# # Part 1: Supplemental Background on Numpy
# 
# This notebook is a quick overview of additional functionality in Numpy. It is intended to supplement the videos and the other parts of this assignment. It does **not** contain any exercises that you need to submit.

# In[ ]:


import sys
print(sys.version)

import numpy as np
print(np.__version__)


# # Random numbers

# Numpy has a rich collection of (pseudo) random number generators. Here is an example; 
# see the documentation for [numpy.random()](https://docs.scipy.org/doc/numpy/reference/routines.random.html) for more details.

# In[ ]:


A = np.random.randint(-10, 10, size=(4, 3)) # return random integers from -10 (inclusive) to 10 (exclusive)
print(A)


# In[ ]:


np.mean(A.T, axis=1)


# # Aggregations or reductions

# Suppose you want to reduce the values of a Numpy array to a smaller number of values. Numpy provides a number of such functions that _aggregate_ values. Examples of aggregations include sums, min/max calculations, and averaging, among others.

# In[ ]:


print("np.max =", np.max(A),"; np.amax =", np.amax(A)) # np.max() and np.amax() are synonyms
print("np.min =",np.min(A),"; np.amin =", np.amin(A)) # same
print("np.sum =",np.sum(A))
print("np.mean =",np.mean(A))
print("np.std =",np.std(A))


# The above examples aggregate over all values. But you can also aggregate along a dimension using the optional `axis` parameter.

# In[ ]:


print("Max in each column:", np.amax(A, axis=0)) # i.e., aggregate along axis 0, the rows, producing column maxes
print("Max in each row:", np.amax(A, axis=1)) # i.e., aggregate along axis 1, the columns, producing row maxes


# # Universal functions

# Universal functions apply a given function _elementwise_ to one or more Numpy objects.
# 
# For instance, `np.abs(A)` takes the absolute value of each element.

# In[ ]:


print(A, "\n==>\n", np.abs(A))


# Some universal functions accept multiple, compatible arguments. For instance, here, we compute the _elementwise maximum_ between two matrices, $A$ and $B$, producing a new matrix $C$ such that $c_{ij} = \max(a_{ij}, b_{ij})$.
# 
# > The matrices must have compatible shapes, which we will elaborate on below when we discuss Numpy's _broadcasting rule_.

# In[ ]:


print(A) # recall A


# In[ ]:


B = np.random.randint(-10, 10, size=A.shape)
print(B)


# In[ ]:


C = np.maximum(A, B) # elementwise comparison
print(C)


# You can also build your own universal functions! For instance, suppose we want a way to compute, elementwise, $f(x) = e^{-x^2}$ and we have a scalar function that can do so:

# In[ ]:


def f(x):
    from math import exp
    return exp(-(x**2))


# This function accepts one input (`x`) and returns a single output. The following will create a new Numpy universal function `f_np`.
# See the documentation for [np.frompyfunc()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frompyfunc.html) for more details.

# In[ ]:


f_np = np.frompyfunc(f, 1, 1)  

print(A, "\n=>\n", f_np(A))


# # Broadcasting

# Sometimes we want to combine operations on Numpy arrays that have different shapes but are _compatible_.
# 
# In the following example, we want to add 3 elementwise to every value in `A`.

# In[ ]:


print(A)
print()
print(A + 3)


# Technically, `A` and `3` have different shapes: the former is a $4 \times 3$ matrix, while the latter is a scalar ($1 \times 1$). However, they are compatible because Numpy knows how to _extend_---or **broadcast**---the value 3 into an equivalent matrix object of the same shape in order to combine them.

# To see a more sophisticated example, suppose each row `A[i, :]` are the coordinates of a data point, and we want to compute the centroid of all the data points (or center-of-mass, if we imagine each point is a unit mass). That's the same as computing the mean coordinate for each column:

# In[ ]:


A_row_means = np.mean(A, axis=0)

print(A, "\n=>\n", A_row_means)


# Now, suppose you want to shift the points so that their mean is zero. Even though they don't have the same shape, Numpy will interpret `A - A_row_means` as precisely this operation, effectively extending or "replicating" `A_row_means` into rows of a matrix of the same shape as `A`, in order to then perform elementwise subtraction.

# In[ ]:


A_row_centered = A - A_row_means
A_row_centered


# Suppose you instead want to mean-center the _columns_ instead of the rows. You could start by computing column means:

# In[ ]:


A_col_means = np.mean(A, axis=1)

print(A, "\n=>\n", A_col_means)


# But the same operation will fail!

# In[ ]:


A - A_col_means # Fails!


# The error reports that these shapes are not compatible. So how can you fix it?
# 
# **Broadcasting rule.** One way is to learn Numpy's convention for **[broadcasting](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting)**. Numpy starts by looking at the shapes of the objects:

# In[ ]:


print(A.shape, A_row_means.shape)


# These are compatible if, starting from _right_ to _left_, the dimensions match **or** one of the dimensions is 1. This convention of moving from right to left is referred to as matching the _trailing dimensions_. In this example, the rightmost dimensions of each object are both 3, so they match. Since `A_row_means` has no more dimensions, it can be replicated to match the remaining dimensions of `A`.

# By contrast, consider the shapes of `A` and `A_col_means`:

# In[ ]:


print(A.shape, A_col_means.shape)


# In this case, per the broadcasting rule, the trailing dimensions of 3 and 4 do not match. Therefore, the broadcast rule fails. To make it work, we need to modify `A_col_means` to have a unit trailing dimension. Use Numpy's [`reshape()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) to convert `A_col_means` into a shape that has an explicit trailing dimension of size 1.

# In[ ]:


A_col_means2 = np.reshape(A_col_means, (len(A_col_means), 1))
print(A_col_means2, "=>", A_col_means2.shape)


# Now the trailing dimension equals 1, so it can be matched against the trailing dimension of `A`. The next dimension is the same between the two objects, so Numpy knows it can replicate accordingly.

# In[ ]:


print("A - A_col_means2\n\n", A, "\n-", A_col_means2) 
print("\n=>\n", A - A_col_means2)


# **Thought exercise.** As a thought exercise, you might see if you can generalize and apply the broadcasting rule to a 3-way array.

# **Fin!** That marks the end of this notebook. If you want to learn more, checkout the second edition of [Python for Data Analysis](http://shop.oreilly.com/product/0636920050896.do) (released in October 2017).

# In[ ]:


pass # Dummy cell

