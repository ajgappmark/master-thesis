'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import euclidean_distances
import collections
import scikit_rp
import matplotlib.pyplot as plt
import csv
import os.path
from numpy.linalg import norm

rows = []
cols = []
values = []
matrix_size = 100
rng = range(0, matrix_size)
for x in rng:
    for y in rng:
        rows.append(x)
        cols.append(y)
        values.append(x)

matrix = csr_matrix( (values,(rows,cols))).todense()
orig_dimension = np.shape(matrix)[1]

print("dataset rows: %s" % np.shape(matrix)[0])
print("dataset columns: %s" % orig_dimension)

# compute the pairwise distances
def getPairwiseDist(matrix):
    dist = {}
    rows = np.shape(matrix)[0]
    count = 0
    for i in range(0, rows):
        for j in range(i+1, rows):
            count = count +1
            entry = int("%s%s" % (i+1,j+1))
            # euclidean distances
            dist[entry] = norm(matrix[i] - matrix[j])
    return count, collections.OrderedDict(sorted(dist.items()))

# perform the evaluation for this dataset for a given dimension
def evaluatePairwiseDistances(dataset, intrinsicDimension):
    # get the random matrix
    rand_matrix = scikit_rp.getSparseRP(intrinsicDimension)._make_random_matrix(intrinsicDimension, orig_dimension)
    # perform the reduction
    reduced_matrix = dataset * rand_matrix.transpose()
    # call method from above
    amount_reduced, reduced_dist = getPairwiseDist(reduced_matrix)

    amount_orig, dist = getPairwiseDist(matrix)
    sumOrig = 0.0
    sumReduced = 0.0
    sumError = 0.0
    keysOrig = dist.iterkeys()
    keysReduced = reduced_dist.iterkeys()
    for keyz in zip(keysOrig, keysReduced):
        orig = keyz[0]
        reduced = keyz[1]

        orig_dist_value = dist[orig]
        reduced_dist_value = reduced_dist[reduced]

        if orig_dist_value > reduced_dist_value:
            error = orig_dist_value - reduced_dist_value
        else:
            error = reduced_dist_value - orig_dist_value

        if orig != reduced:
            raise "error. '%s' must be equal to '%s'" % (orig, reduced)

        sumOrig = sumOrig + orig_dist_value
        sumReduced = sumReduced + reduced_dist_value
        sumError = sumError + error

    return (sumOrig, sumReduced, sumError)

avg_error = 0.0
folds = range(0, 30)
x = []
y_orig = []
y_reduced = []
y_error = []
for dimension in np.arange(5, 16, 1):
    print "dimension %s" % dimension
    x.append(dimension)
    error_sum = []
    orig_sum = []
    reduced_sum = []
    for i in folds:
        orig, reduced, error = evaluatePairwiseDistances(matrix, dimension)
        error_sum.append(error)
        orig_sum.append(orig)
        reduced_sum.append(reduced)
        # print error_sum
    y_orig.append(np.mean(orig_sum))
    y_reduced.append(np.mean(reduced_sum))
    y_error.append(np.mean(error_sum))

outputFolder = os.path.dirname(os.path.abspath(__file__))
outputFolder = "%s/csv" % outputFolder

with open("%s/result_python_%s.csv" % (outputFolder, matrix_size), "wb") as csvfile:
    writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["x", "y_orig","y_reduced" ,"y_error"])
    for item in zip(x,y_orig, y_reduced, y_reduced):
        writer.writerow([item[0], item[1], item[2], item[3]])


'''
plt.xlabel("iteration")
plt.ylabel("error")
plt.grid()

plt.plot(x, y, label="error of pairwise distances")

plt.legend(loc="best")
plt.savefig("output/pairwise/pairwise_python.png", dpi=320)

print "mean error %s" % np.mean(y)
'''