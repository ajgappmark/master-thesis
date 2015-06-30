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

def rocAuc(trainData, trainLabel, testData, testLabel):

    train = list()
    test = list()

    for k in range(0, 5):
        regr = linear_model.LogisticRegression()
        regr.fit(trainData, trainLabel)

        trainScores = regr.score(trainData, trainLabel)
        testScores = regr.score(testData, testLabel)

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

def getLearningCurve(data, label):
    #maxItemsInDataset = len(label)

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

        #trainRoc_auc = metrics.roc_auc_score(trainLabel, trainScores)
        #testRoc_auc = metrics.roc_auc_score(testLabel, testScores)

        #yTrain.append(trainRoc_auc)
        #yTest.append(testRoc_auc)

        if hasattr(trainData[0], "indices"):
            numInstances = np.shape(trainData[0])
            numInstances = numInstances[0]
        else:
            numInstances = np.shape(trainData)
            numInstances = numInstances[1]
        # print numInstances
        #xPercentage = (numInstances[0] * 100) / maxItemsInDataset
        x.append(numInstances)
        #y.append(np.mean(scores))

    return x, yTrain, yTest

dataSets = data_factory.getAllDatasets()
for i in range(len(dataSets)):

    plt.figure(i)

    load = dataSets[i]
    data, label, desc, size = load()

    '''
    if size > 0:
        initialReduceBlockSize = np.arange(size, size+0.1, 0.1)
        trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, testSetPercentage)

        data = trainDataBlocks[0][0]
        label = trainLabelBlocks[0][0]
    '''

    print np.shape(data)
    print np.shape(label)

    plt.subplot(111)

    plt.title("%s - learning curve" % (desc))
    plt.xlabel("size of dataset")
    plt.ylabel("score")

    # plt.ylim([0.98, 1.0])

    plt.grid()

    # not reduced dataset
    x, yTrain, yTest = getLearningCurve(data, label)
    plt.plot(x, yTest, label="test")
    plt.plot(x, yTrain, label="train")



    plt.legend(loc="best")
    plt.savefig("output_new/curve_%s.png" % (desc), dpi=320)
