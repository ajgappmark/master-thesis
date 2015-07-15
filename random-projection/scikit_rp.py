'''
Author: Sebastian Alfers
This file is part of the master thesis about Dimensionality Reduction
'''

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection


def getGaussianRP(new_dimension):
    return GaussianRandomProjection(n_components=new_dimension)

# scikit-learn implementation: gaussian matrix
def gaussianRP(data, new_dimension):
    rp = getGaussianRP(new_dimension)
    return rp.fit_transform(data)

# scikit-learn implementation: sparse matrix
def sparseRP(data, new_dimension):
    rp = SparseRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)
