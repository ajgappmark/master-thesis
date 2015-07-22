from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import euclidean_distances
import collections
import scikit_rp

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

print("orig rows: %s" % np.shape(matrix)[0])
print("orig dimension: %s" % orig_dimension)
print matrix

def getPairwiseDist(matrix):
    dist = {}
    rows = np.shape(matrix)[0]
    for i in range(0, rows):
        for j in range(i+1, rows):
            entry = int("%s%s" % (i+1,j+1))
            dist[entry] = norm(matrix[i] - matrix[j])
            #print "from %s to %s = %s" % (matrix[i], matrix[j], dist[entry])
    return collections.OrderedDict(sorted(dist.items()))

dist = getPairwiseDist(matrix)

new_dimension = 5


rand_matrix = scikit_rp.getSparseRP(new_dimension)._make_random_matrix(new_dimension, orig_dimension)
print np.shape(rand_matrix)
reduced_matrix = matrix * rand_matrix.transpose()
reduced_dist = getPairwiseDist(reduced_matrix)

print "reduced hape:"

keysOrig = dist.iterkeys()
keysReduced = reduced_dist.iterkeys()
for keyz in zip(keysOrig, keysReduced):
    a = keyz[0]
    b = keyz[1]
    if a != b:
        raise "error. '%s' must be equal to '5s'" % (a, b)

    if dist[a] > reduced_dist[b]:
        d = dist[a] - reduced_dist[b]
    else:
        d = reduced_dist[b] - dist[a]

    print "#%s orig: %s | reducd: %s => %s " % (a, dist[a], reduced_dist[b], d)




#print "first pairwise dist amount of results: %s" % np.shape(dist)[0]

#print dist

