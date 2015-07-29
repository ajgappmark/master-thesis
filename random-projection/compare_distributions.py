'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import matplotlib.pyplot as plt
import os.path
import distributions
import build_rp_matrix as rp
import scikit_rp
import numpy as np

rows = 570
columns = 3000

dist = distributions.dense_2
matrixA = rp.buildMatrix(rows, columns, dist)

matrixB = scikit_rp.getGaussianRP(columns)._make_random_matrix(rows, columns)

print np.shape(matrixA)
print np.shape(matrixB)

def buildHistogram(data):
    buckets = dict()
    for row in data:
        for value in row:

            bucket = "%.2f" % float(value)
            #print bucket
            if float(bucket) in buckets:
                buckets[float(bucket)] += 1
            else:
                #print float(bucket)
                buckets[float(bucket)] = 1
    return buckets

histA = buildHistogram(matrixB)
histB = buildHistogram(matrixB)

plt.subplot(211)
plt.bar(list(histA.iterkeys()), list(histA.values()), 0.01, color="black")
plt.grid()
plt.title("histogram of python's random.randrange()")
plt.xlabel("bins")
plt.ylabel("amount of hits / bin")


plt.subplot(212)
plt.bar(list(histB.iterkeys()), list(histB.values()), 0.01, color="black")
plt.grid()
plt.title("histogram of python's random.randrange()")
plt.xlabel("bins")
plt.ylabel("amount of hits / bin")



folder = os.path.dirname(os.path.abspath(__file__))
plt.savefig("%s/output/distributions_histogram.png" % folder, dpi=320)
