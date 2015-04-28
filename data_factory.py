from sklearn import cross_validation
import cancer_datasets as cancerDataSets
import numpy as np
import convert as plistaDataSets

def loadFirstCancerDataset():
    data, label = cancerDataSets.loadFirst()
    return data, label, "cancer 1", 0


def loadSecondCancerDataset():
    data,label = cancerDataSets.loadSecond()
    return data, label, "cancer 2", 0

def loadFirstPlistaDataset(size = 0):
    data,label = plistaDataSets.loadFirst()
    return data, label, "impressions 1", size

def loadSecondPlistaDataset():
    data,label = plistaDataSets.loadSecond()
    return data, label, "impressions 2", 0

def loadThridPlistaDataset():
    data,label = plistaDataSets.loadThird()
    return data, label, "impressions 3", 0

def loadFourthPlistaDataset():
    data,label = plistaDataSets.loadFirth()
    return data, label, "impressions 4", 0

def loadFifthPlistaDataset(size = 0):
    data, label = plistaDataSets.loadFifth()
    return data, label, "impressions 5", size

def loadSixthPlistaDataset(size = 0):
    data, label = plistaDataSets.loadSixth()
    return data, label, "impressions 6", size

def loadBigDataset():
    data,label = plistaDataSets.loadBig()
    return data, label, "big dataset", 0

def splitDatasetInBlocks(data, labels, trainBlockSizes, testSetPercentage):

    trainDataBlocks = []
    trainLabelBlocks = []
    testDataBlocks = []
    testLabelBlocks = []

    for i in range(len(trainBlockSizes)):
        train = trainBlockSizes[i]
        test = testSetPercentage * trainBlockSizes[i]

        skf = cross_validation.StratifiedShuffleSplit(labels, 5, train_size=train, test_size=test)

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


def getSmallDatasets():
    sets = [loadFifthPlistaDataset, loadFirstPlistaDataset]
    return sets

def getAllDatasets():
    dataSets = [ loadFirstCancerDataset,
                 loadSecondCancerDataset,
                 loadFirstPlistaDataset,
                 loadSecondPlistaDataset,
                 loadThridPlistaDataset,
                 loadFourthPlistaDataset
               ]
    return dataSets

def getExperimentDataset():
    dataSets = [ loadBigDataset ]
    return dataSets

def getAllPlistaDatasets():
    dataSets = [ loadFirstPlistaDataset,
                 loadSecondPlistaDataset,
                 loadThridPlistaDataset,
                 loadFourthPlistaDataset
            ]
    return dataSets

def getAllCancerDatasets():
    dataSets = [ loadFirstCancerDataset,
                 loadSecondCancerDataset
            ]
    return dataSets