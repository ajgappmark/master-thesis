import numpy as np
import data_factory
from analyze import analyze
import dr
from sklearn import cross_validation, linear_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dimensions = np.arange(50, 300, 40)
algos = ["hash", "rp", "incremental_pca"]
testSetPercentage = 0.1

def reduceForDimension(algo, data, label):
    # now do reduction and plot the performance based on the reduced data

    x = list()
    y = list()
    for dimension in dimensions:
        x.append(dimension)

        lr = linear_model.LogisticRegression()
        reduced ,duration = dr.reduceByKey(algo, data, label, dimension)
        scores = cross_validation.cross_val_score(lr, reduced, label, scoring='roc_auc')
        y.append(scores.mean())

    return x, y

dataSets = data_factory.getSmallDatasets()
for i in range(len(dataSets)):
    load = dataSets[i]
    data, label, desc, size = load(0.1)

    print "original data"
    analyze(data, label)

    if size > 0:
        initialReduceBlockSize = np.arange(size, size+0.2, 0.1)
        trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, testSetPercentage)

        data = trainDataBlocks[0][0]
        label = trainLabelBlocks[0][0]

    print np.shape(data)
    print np.shape(label)
    analyze(data, label)

    plt.subplot(111)
    plt.figure(i)
    plt.title("%s - different dimensions" % (desc))
    plt.xlabel("reduced dimensions")
    plt.ylabel("roc auc score")

    # plt.ylim([0.98, 1.0])

    plt.grid()

    # not reduced dataset

    for algo in algos:
        x, y = reduceForDimension(algo, data, label)
        plt.plot(x, y, label=algo)

    plt.legend(loc="best")

    plt.savefig("output/compare_dimensions/compare_dimensions_%s.png" % (desc), dpi=320)
