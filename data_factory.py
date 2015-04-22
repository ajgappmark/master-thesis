from sklearn import cross_validation
import cancer_datasets as cancerDataSets
import numpy as np
import convert as plistaDataSets

def loadFirstCancerDataset():
    data, label = cancerDataSets.loadFirst()
    return data, label, "cancer 1"


def loadSecondCancerDataset():
    data,label = cancerDataSets.loadSecond()
    return data, label, "cancer 2"

def loadFirstPlistaDataset():
    data,label = plistaDataSets.loadFirst()
    return data, label, "impressions 1"

def loadSecondPlistaDataset():
    data,label = plistaDataSets.loadSecond()
    return data, label, "impressions 2"

def loadThridPlistaDataset():
    data,label = plistaDataSets.loadThird()
    return data, label, "impressions 3"

def loadBigDataset():
    data,label = plistaDataSets.loadBig()
    return data, label, "big dataset"

def splitDatasetInBlocks(data, labels, trainBlockSizes, testSetPercentage):

    trainDataBlocks = []
    trainLabelBlocks = []
    testDataBlocks = []
    testLabelBlocks = []

    for i in range(len(trainBlockSizes)):
        train = trainBlockSizes[i]
        test = testSetPercentage * trainBlockSizes[i]

        skf = cross_validation.StratifiedShuffleSplit(labels, 2, train_size=train, test_size=test)

        a = []
        b = []
        c = []
        d = []

        for trainIndex, testIndex in skf:
            a.append(data[trainIndex])
            b.append(labels[trainIndex])
            c.append(data[testIndex])
            d.append(labels[testIndex])

        trainDataBlocks.append(a)
        trainLabelBlocks.append(b)
        testDataBlocks.append(c)
        testLabelBlocks.append(d)

    return trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks

def getAllDatasets():
    dataSets = [ loadFirstCancerDataset,
                 loadSecondCancerDataset,
                 loadFirstPlistaDataset,
                 loadSecondPlistaDataset,
                 loadThridPlistaDataset
            ]
    return dataSets

def getExperimentDataset():
    dataSets = [ loadBigDataset ]
    return dataSets

def getAllPlistaDatasets():
    dataSets = [ loadFirstPlistaDataset,
                 loadSecondPlistaDataset,
                 loadThridPlistaDataset
            ]
    return dataSets

def getAllCancerDatasets():
    dataSets = [ loadFirstCancerDataset,
                 loadSecondCancerDataset
            ]
    return dataSets