from sklearn import cross_validation
import cancer_datasets as cancer
from sklearn import linear_model
import convert
import numpy as np
import analyze

data, labels = cancer.loadFirstDataset()

def printAccuracy(data, labels):
    print "shape data"
    print np.shape(data)

    print "shape labels"
    print np.shape(labels)

    score = "accuracy"
    regr = linear_model.LogisticRegression()
    score = cross_validation.cross_val_score(regr, data, labels, scoring=score).mean()

    print score



def splitInBlocks(data, labels, trainBlockSizes, testSetPercentage):

    print trainBlockSizes

    trainDataBlocks = []
    trainLabelBlocks = []
    testDataBlocks = []
    testLabelBlocks = []

    for i in range(len(trainBlockSizes)):
        train = trainBlockSizes[i]
        print train
        test = testSetPercentage * trainBlockSizes[i]

        skf = cross_validation.StratifiedShuffleSplit(labels, 5, train_size=train, test_size=test)

        a = []
        b = []
        c = []
        d = []

        for trainIndex, testIndex in skf:
            #print "train: %s data, %s labels" % (len(data[trainIndex]), len(labels[trainIndex]))
            #print "test: %s data, %s labels" % (len(data[testIndex]), len(labels[testIndex]))
            a.append(data[trainIndex])
            b.append(labels[trainIndex])
            c.append(data[testIndex])
            d.append(labels[testIndex])

        trainDataBlocks.append(a)
        trainLabelBlocks.append(b)
        testDataBlocks.append(c)
        testLabelBlocks.append(d)

        #print "rows: %s, labels: %s -> clicks: %s, ratio: %s " % (len(trainData), len(trainLabels), trainLabels.sum(), trainLabels.sum()*100/len(t
    #print iter(skf).next()
    #print trainDataBlocks[0]rainData))

    '''
        trainIndex, testIndex = iter(skf).next()

        trainDataBlocks.append(data[trainIndex])
        trainLabelBlocks.append(labels[trainIndex])

        testDataBlocks.append(data[testIndex])
        testLabelBlocks.append(labels[testIndex])
    '''


    return trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks

print "--------- analyze base dataset --------------"
analyze.analyze(data, labels)
print "--------- finished analyze base dataset --------------"

data, labels = cancer.loadFirstDataset()
#printAccuracy(data, labels)
trainBlockSizes = np.arange(0.1, 0.6, 0.1)
testSetPercentage = 0.2

trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = splitInBlocks(data, labels, trainBlockSizes, testSetPercentage)

print "trainDataBlocks"
print np.shape(trainDataBlocks)

print "trainLabelBlocks"
print np.shape(trainLabelBlocks)

print "testDataBlocks"
print np.shape(testDataBlocks)

print "testLabelBlocks"
print np.shape(testLabelBlocks)

for i in range(len(trainDataBlocks)):


    data = trainDataBlocks[i][0]
    label = trainLabelBlocks[i][0]
    print "ok"
    print np.shape(data)
    print np.shape(label)

    analyze.analyze(data, label)
    print "-------------------------------------"

'''
analyze.analyze(data,labels)

# skf = cross_validation.StratifiedKFold(labels, n_folds=20)
# skf = cross_validation.StratifiedShuffleSplit(labels, 3, test_size=0.1, random_state=0)

print len(skf)

for train_index, test_index in skf:
    print '## split start ##'
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print "train:"
    print np.shape(X_train)
    print np.shape(y_train)
    print "test:"
    print np.shape(X_test)
    print np.shape(y_train)

    analyze.analyze(X_train,y_train)
    analyze.analyze(X_test,y_test)
    print '## split end ##'


matrix, labels = convert.convert("../plista-data/export_sparse_publisher_vectors_11_1477_hour_3-1.csv")
printAccuracy(matrix, labels)

matrix, labels = convert.convert("../plista-data/export_sparse_publisher_vectors_11_13725,970,4787_hour_2-1.csv")
printAccuracy(matrix, labels)
'''