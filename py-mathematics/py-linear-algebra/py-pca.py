'''
Principal Component Analysis (PCA) is a method for reducing the dimensionality of data (i,e ...) transforming a 10-D data into 2-D data, without loosing the esscence of the original data.

It can be imagined as a projection method where data with m-columns (features) is projected into a subspace with <= m columns.
'''

import numpy as np
from numpy import array, mean, cov
from numpy.linalg import eig
from sklearn.decomposition import PCA

print("\nNumpy Version:\t", np.__version__)

matrix_a = np.array([[0, 3, 6, 10], [10,13,16,20],[20,23,26,30],[30,33,36,40]])
print("\nMatrix A:\n", matrix_a)

'''
PCA Algorithm Steps:
    1. Let A be a matrix (nxm)
    2. Let M be mean of Matrix A. M = mean(A)
    3. Center the values of A by subtracting the mean column value. C = A-M
    4. Find the Covariance Matrix of the recentered Matrix. CovM = covariance(C)
    5. Find the eigen values and eigen vectors of Covariance Matrix. eigvals, eigvectors = eig(CovM)
        - Eigen Vectors represents the direction or componsents for the reduced subspace.
        - Eigen Values represents the magnitudes of the directions.
        - Eigen Value at Index 0 is eigen value for Eigen Vector at index 0.
    6. Sort Eigen Values in descending order to provide a ranking for components or axes.
        - If Eigen Values are sorted, then eigen vectors should also be sorted by sorted eigen values.
    7. Choose Eigen Values and Eigen Vectors as Principal Components of the Original Matrix.
        Condition to choose:
            - If all the values in Eigen Values are close to each other, then we have reached the end of decomposition as further decomposition will not reduce the number of axis.
            - If the eigen values are close to zero, then discard them.
            - Select K eigen-vectors, called Principal Components, that have the k-largest eigen-values.
        B = select(eigen_values, eigen_vectors)
    8. Once choose, project the principal components into subspace using matrix multiplication.
        PCA = B.T*A
    
'''
mean_a = mean(matrix_a)
print("\nMean of Matrix A:\n", mean_a)

recentered_matrix_a = matrix_a - mean_a
print("\nRecenterd Matrix A:\n", recentered_matrix_a)

covarince_matrix = cov(recentered_matrix_a)
print("\nCovariance Matrix:\n", covarince_matrix)

eigen_values, eigen_vectors = eig(covarince_matrix)
print("\nEigen Values:\n{0}\nEigen Vectors:\n{1}\n".format(eigen_values, eigen_vectors))

pca = eigen_vectors.T.dot(recentered_matrix_a.T)
print("\nProjected Principal Components:\n", pca.T)


'''
Using scikit-learn PCA transformation method
'''
# Initialize PCA and reduce the number of components to 2
pca_sk = PCA(4)

# Apply the PCA onto the parent matrix.
pca_sk.fit(matrix_a)
print("\nPrincipal Components from Scikit:\n",pca_sk.components_, "\nVariance:\n", pca_sk.explained_variance_)

# Transform Original Data to PCA
transformed_data = pca_sk.transform(matrix_a)
print("\nTransformed Data:\n", transformed_data)