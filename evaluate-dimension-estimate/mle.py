'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold, datasets
import data_factory as df
from analyze import analyze
from sklearn.preprocessing import OneHotEncoder
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''
    - this is a implementation of the papter "Maximum Likelihood Estimation of Intristic Dimension"
    - it can be found at: https://www.stat.berkeley.edu/~bickel/mldim.pdf
    - last access: 28.4.2015
'''


def mle(X):
    neigh = NearestNeighbors()
    neigh.fit(X)

    if hasattr(X, "shape"):
        numNeighbors = X.shape
        numNeighbors = numNeighbors[0]
    else:
        numNeighbors = len(X)

    # distance from x to k-th NN
    def KthDistaneFromx(k, distanceFrom):
        distances, neighbors = neigh.kneighbors(distanceFrom, numNeighbors, return_distance=True)

        # skip first item since it is the item itself
        distances = distances[:,1:]
        #neighbors = neighbors[:,1:]

        #if distances[0][k] == 0.0:
        #    print "dist is %s" % (distances[0][k])

        # correct?
        return distances[0][k]

    def mHutKX(k, Xi):
        sum = 0

        # maybe k-1 ???
        for j in range(k-1):
            nominator = KthDistaneFromx(k, Xi)
            denominator = KthDistaneFromx(j, Xi)

            if denominator > 0 and denominator > 0.0:
                result = nominator / denominator
                sum = sum + np.log(result)
            '''
                test = 3
                nominator = KthDistaneFromx(k, Xi)
                denominator = KthDistaneFromx(j, Xi)
            '''

        ret = (1.0 / (k -1)) * sum
        if ret == 0.0:
            return 0
        else:
            return 1.0 / ret

    def mHutK(k):
        sum = 0

        for i in range(numNeighbors):
            #if i % 500 == 0:
                #print "%s -> %s" % (numNeighbors, i)
            sum = sum + mHutKX(k, X[i])
            #print sum

        ret = (1.0 / numNeighbors ) * sum
        return ret

    def hHut():
        sum = 0
        k1 = 40
        k2 = 60
        stepsize = 1
        print 'start range'
        for k in np.arange(k1, k2, stepsize):

            r = mHutK(k)
            if r % 500 == 0:
                print " %s -> %s" % (k, r)
            #print sum
            sum = sum + r

        denominator = k2 - k1 + 1
        print denominator
        return (1.0 / denominator)*sum


    '''
    sel = VarianceThreshold(threshold=0.1)
    print np.shape(X)
    print np.shape(sel.fit_transform(X))
    '''

    #print hHut()

    start = time.time()
    estimatedDimension = mHutK(5)
    end = time.time()
    return (estimatedDimension, end - start)

'''
datasets = df.getAllCancerDatasets()
for load in datasets:
    data, label, desc, size = load()
    print "experiments with dataset: %s" % desc
    analyze(data, label, "before binary encode")
    estimatedDimension = mle(data)
    print "estimated dimension without binary encoding: %s" % estimatedDimension
    enc = OneHotEncoder()
    enc.fit(data)
    data = enc.transform(data).toarray()
    analyze(data, label, "after binary encode")
    estimatedDimensionWithBinaryEncode = mle(data)
    print "estimated dimension with binary encoding: %s" % estimatedDimensionWithBinaryEncode
    print
'''

data, label, desc, size = df.loadFirstPlistaDataset()

trainBlockSizes = np.arange(0.001, 0.005, 0.001)
testSetPercentage = 0.2
trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = df.splitDatasetInBlocks(data, np.array(label), trainBlockSizes, testSetPercentage)

# estimate for different sizes of the data set
x = list()
yDuration = list()
yDimension = list()
for i in range(len(trainDataBlocks)):
    estimatedDimension = list()
    duration = list()

    shape = np.shape(trainDataBlocks[i][0])
    x.append(shape[0])

    for j in range(len(trainDataBlocks[i])):
        print j
        data = trainDataBlocks[i][j]
        label = trainLabelBlocks[i][0]

        e, d = mle(data)
        estimatedDimension.append(e)
        duration.append(d)



    #analyze(data, label)


    print "estimated dimension: %s" % estimatedDimension

    yDuration.append(np.mean(duration))
    yDimension.append(np.mean(estimatedDimension))



plt.subplot(211)
plt.title("Evaluation of Dimension Estimator")
plt.xlabel("size of dataset")
plt.ylabel("duration of algorithm")

plt.grid()

plt.plot(x, yDuration)

plt.subplot(212)
plt.xlabel("size of dataset")
plt.ylabel("estimated dimension")

plt.grid()

plt.plot(x, yDimension)

folder = os.path.dirname(os.path.abspath(__file__))
plt.savefig("%s/output/evaluate_dimension_estimator.png"  % folder, dpi=320)

