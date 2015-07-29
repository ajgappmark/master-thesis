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
rng = range(0, 10)
for x in rng:
    for y in rng:
        rows.append(x)
        cols.append(y)
        values.append(x)

matrix = csr_matrix( (values,(rows,cols))).todense()

orig_dimension = np.shape(matrix)[1]

print("dataset rows: %s" % np.shape(matrix)[0])
print("dataset columns: %s" % orig_dimension)

def getPairwiseDist(matrix):
    dist = {}
    rows = np.shape(matrix)[0]
    count = 0
    for i in range(0, rows):
        for j in range(i+1, rows):
            count = count +1
            entry = int("%s%s" % (i+1,j+1))
            dist[entry] = norm(matrix[i] - matrix[j])
            #print "from %s to %s = %s" % (matrix[i], matrix[j], dist[entry])
    return count, collections.OrderedDict(sorted(dist.items()))

amount_orig, dist = getPairwiseDist(matrix)

new_dimension = 5


def evaluatePairwiseDistances(dataset, intrinsicDimension):

    rand_matrix = scikit_rp.getSparseRP(intrinsicDimension)._make_random_matrix(intrinsicDimension, orig_dimension)
    reduced_matrix = dataset * rand_matrix.transpose()
    amount_reduced, reduced_dist = getPairwiseDist(reduced_matrix)

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

        #print "sum orig of %s -> %s" % (orig, sumOrig + orig_dist_value)

        #print "#%s orig: %s | reducd: %s, error: %s " % (orig, orig_dist_value, reduced_dist_value, error)
        sumOrig = sumOrig + orig_dist_value
        sumReduced = sumReduced + reduced_dist_value
        sumError = sumError + error

    #print amount_reduced

    #print "sum of original distances: %s" % sumOrig
    #print "sum of reduced distances: %s" % sumReduced
    #print "sum of error distances: %s" % sumError
    return (sumOrig, sumReduced, sumError)
    #print "first pairwise dist amount of results: %s" % np.shape(dist)[0]

    #print dist

avg_error = 0.0
folds = range(0, 200)

x = []
y = []
for dimension in np.arange(5, 16, 1):
    print "dimension %s" % dimension
    x.append(dimension)
    error_sum = []
    for i in folds:
        orig, reduce, error = evaluatePairwiseDistances(matrix, new_dimension)
        error_sum.append(error)
        # print error_sum
    y.append(np.mean(error_sum))

outputFolder = os.path.dirname(os.path.abspath(__file__))
outputFolder = "%s/csv" % outputFolder

with open("%s/result_python.csv" % outputFolder, "wb") as csvfile:
    writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["x", "y"])
    for item in zip(x,y):
        writer.writerow([item[0], item[1]])

    writer.writerow(["mean", np.mean(y)])
'''
plt.xlabel("iteration")
plt.ylabel("error")
plt.grid()

plt.plot(x, y, label="error of pairwise distances")

plt.legend(loc="best")
plt.savefig("output/pairwise/pairwise_python.png", dpi=320)

print "mean error %s" % np.mean(y)
'''