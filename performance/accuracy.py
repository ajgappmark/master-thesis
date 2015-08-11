'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import linear_model
import data_factory as factory
import numpy as np
from analyze import analyze

def calcScore(metric, data, labels):
    regr = linear_model.LogisticRegression()
    score = cross_validation.cross_val_score(regr, data, labels, scoring=metric).mean()
    return score



def drawGraphForDatasets(datasets, fileName, item, trainBlockSizes, metric, ylim = []):

    plt.subplot(111)
    plt.figure(item)
    plt.title(metric)
    plt.xlabel("% of dataset")
    plt.ylabel("score: %s" % metric)

    plt.grid()

    for load in datasets:
        # load it lazy
        data, label, desc = load()

        # for test - make dataset smaller
        initialReduceBlockSize = np.arange(0.5, 0.7, 0.1)
        trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = factory.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, testSetPercentage)
        data = trainDataBlocks[0][0]
        label = trainLabelBlocks[0][0]

        analyze(data, label)

        maxItemsInDataset = len(label)

        testSetPercentage = 0.02

        trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = factory.splitDatasetInBlocks(data, np.array(label), trainBlockSizes, testSetPercentage)

        x = list()
        y = list()

        for i in range(len(trainDataBlocks)):
            trainData = trainDataBlocks[i]
            trainLabel = trainLabelBlocks[i]
            # testData = testDataBlocks[i]
            # testLabel = testLabelBlocks[i]

            numInstances = np.shape(trainData[0])
            score = calcScore(metric, trainData[0], trainLabel[0])


            xPercentage = (numInstances[0] * 100) / maxItemsInDataset
            x.append(xPercentage)

            #y.append(float("%.4f" % score))
            y.append(score)
            #print "x:%s, y:%s" % (numInstances[0], score)
        print "------------------------"
        print y
        print np.mean(y)
        print "------------------------"

        plt.plot(x, y, label=desc)


    plt.legend(loc="best")
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.savefig("performance/output/%s_%s.png" % (fileName, metric), dpi=320)


trainBlockSizesPlista = np.arange(0.2, 0.8, 0.05)
drawGraphForDatasets(factory.getAllPlistaDatasets(), "plista", 1, trainBlockSizesPlista, "roc_auc")

trainBlockSizesCancer = np.arange(0.1, 0.6, 0.01)
drawGraphForDatasets(factory.getAllCancerDatasets(), "cancer", 0, trainBlockSizesCancer, "roc_auc")

#drawGraphForDatasets(factory.getExperimentDataset(), "experiment", 0, trainBlockSizesPlista, "roc_auc")
