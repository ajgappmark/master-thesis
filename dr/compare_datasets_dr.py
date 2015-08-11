'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import data_factory
from sklearn import cross_validation

from sklearn import linear_model
import numpy as np
import dr
from analyze import analyze

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

metric = "roc_auc"
trainBlockSizes = np.arange(0.1, 0.9, 0.1)
testSetPercentage = 0.02

# build the roc auc value
def rocAuc(data, labels):
    regr = linear_model.LogisticRegression()
    score = cross_validation.cross_val_score(regr, data, labels, scoring=metric).mean()
    return score

# build the
def getXYForDataLabel(data, label):
    maxItemsInDataset = len(label)

    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), trainBlockSizes, testSetPercentage)
    x = list()
    y = list()
    for i in range(len(trainDataBlocks)):
        trainData = trainDataBlocks[i]
        trainLabel = trainLabelBlocks[i]

        scores = list()
        for j in range(len(trainData)):
            s = rocAuc(trainData[j], trainLabel[j])
            scores.append(s)

        numInstances = np.shape(trainData[0])
        xPercentage = (numInstances[0] * 100) / maxItemsInDataset
        x.append(xPercentage)
        y.append(np.mean(scores))

    return x, y

dataSets = data_factory.getSmallDatasets()
for i in range(len(dataSets)):
    load = dataSets[i]
    data, label, desc, size = load()

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
    plt.title("%s - %s" % (desc, metric))
    plt.xlabel("% of dataset")
    plt.ylabel(metric)

    plt.grid()

    # not reduced dataset
    x, y = getXYForDataLabel(data, label)
    plt.plot(x, y, label="not reduced")

    # now do reduction and plot the performance based on the reduced data
    algos = ["hash", "rp", "incremental_pca"]

    for algo in algos:
        reduced ,duration = dr.reduceByKey(algo, data, label, 5)
        x, y = getXYForDataLabel(reduced, label)
        plt.plot(x, y, label=algo)


    plt.legend(loc="best")
    plt.savefig("dr/output/%s.png" % (desc), dpi=320)
