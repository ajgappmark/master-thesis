'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import data_factory
from sklearn import cross_validation

from sklearn import metrics, linear_model
import numpy as np
import dr
from analyze import analyze

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

metric = "roc_auc"
trainBlockSizes = np.arange(0.01, 0.8, 0.05)
testSetPercentage = 0.2

# perform LR and compute ROC AUC based on test data
def rocAuc(trainData, trainLabel, testData, testLabel):

    train = list()
    test = list()

    for k in range(0, 5):
        regr = linear_model.LogisticRegression()
        regr.fit(trainData, trainLabel)

        trainProba = regr.predict_proba(trainData)
        testProba = regr.predict_proba(testData)

        trainPosivieLabelProba = trainProba[:, 1]
        testPosivieLabelProba = testProba[:, 1]

        a = np.array(trainLabel)
        b = np.array(trainPosivieLabelProba)

        c = np.array(testLabel)
        d = np.array(testPosivieLabelProba)

        trainRoc_auc = metrics.roc_auc_score(a, b)
        testRoc_auc = metrics.roc_auc_score(c, d)

        train.append(trainRoc_auc)
        test.append(testRoc_auc)

    return np.mean(train), np.mean(test)

# build learning curve
def getLearningCurve(data, label):
    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), trainBlockSizes, testSetPercentage)
    x = list()
    yTrain = list()
    yTest = list()
    for i in range(len(trainDataBlocks)):
        trainData = trainDataBlocks[i]
        trainLabel = trainLabelBlocks[i]
        testData = testDataBlocks[i]
        testLabel = testLabelBlocks[i]

        trainScores = list()
        testScores = list()
        for j in range(len(trainData)):
             trainS, testS = rocAuc(trainData[j], trainLabel[j], testData[j], testLabel[j])
             trainScores.append(trainS)
             testScores.append(testS)

        yTrain.append(np.mean(trainScores))
        yTest.append(np.mean(testScores))

        if hasattr(trainData[0], "indices"):
            numInstances = np.shape(trainData[0])
            numInstances = numInstances[0]
        else:
            numInstances = np.shape(trainData)
            numInstances = numInstances[1]

        x.append(numInstances)

    return x, yTrain, yTest

dataSets = data_factory.getAllDatasets()
for i in range(len(dataSets)):
    plt.figure(i)
    load = dataSets[i]
    data, label, desc, size = load()

    print np.shape(data)
    print np.shape(label)

    plt.subplot(111)

    plt.title("%s - learning curve" % (desc))
    plt.xlabel("size of dataset")
    plt.ylabel("score")

    plt.grid()

    # not reduced dataset
    x, yTrain, yTest = getLearningCurve(data, label)
    plt.plot(x, yTest, label="test")
    plt.plot(x, yTrain, label="train")

    plt.legend(loc="best")
    plt.savefig("output_new/curve_%s.png" % (desc), dpi=320)
