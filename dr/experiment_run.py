import numpy as np
import data_factory
from analyze import analyze
import dr
from sklearn import cross_validation, linear_model
from sklearn.preprocessing import OneHotEncoder
import os.path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time




# reduce the data by given algo
def durationRocAuc(algo, data, label, dimension):
    reduced ,duration = dr.reduceByKey(algo, data, label, dimension)

    lr = linear_model.LogisticRegression()
    score = cross_validation.cross_val_score(lr, reduced, label, scoring='roc_auc')

    return duration, reduced, score.mean()

def measureFitLR(data, label):

    # measure for 10 times and avg
    sum = list()
    for i in range(0, 10):
        print i

        start = time.time()
        lr = linear_model.LogisticRegression()
        lr.fit(data, label)
        end = time.time()
        sum.append(end - start)
    print "-----"
    print sum
    print np.mean(sum)
    print
    return np.mean(sum)


#testSetPercentage = 0.1

#data, label, desc, size = data_factory.loadSixthPlistaDataset(0.02)
#data, label, desc, size = data_factory.loadFirstCancerDataset()

#print "-----original set------"
#analyze(data, label)


#if size > 0:
#    initialReduceBlockSize = np.arange(size, size+0.2, 0.1)
#    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, testSetPercentage)

#    data = trainDataBlocks[0][0]
#    label = trainLabelBlocks[0][0]

#print "-----small set------"
#analyze(data, label)

#print "NO dr lr took: %f" % measureFitLR(data, label)


#'''

def execute(experiment):
    folder = setupExperimentFolder(experiment)
    algos = experiment['algos']
    metrics = experiment['yValues']
    dimensions = experiment["dimensions"]
    experimentName = experiment["name"]

    data, label, description, reduce = loadData(experiment)

    analyze(data, label)

    # we want one figure for each y-metric
    x, yValues = runExperimentForMetric(data, label, algos, dimensions)
    for i in range(len(metrics)):
        metric = metrics[i]
        plt.figure(i)
        plt.subplot(111)
        plt.grid()
        plt.xlabel("dimensions")
        plt.ylabel(metric)

        for algo in yValues.iterkeys():

            y = yValues[algo][metric]
            lbl = "%s - (%.2f)" % (algo, np.mean(y))
            #print "*******"
            #print(lbl)
            plt.plot(x, y, label=lbl)

        #plt.legend(loc="best")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   fancybox=True, shadow=True, ncol=2)

        plt.savefig("%s/dimension_vs_%s.png" % (folder, metric), dpi=320, bbox_inches = "tight")

        with open("%s/log_dimension_vs_%s.csv" % (folder, metric), "wb") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            x = [str(i) for i in x]
            writer.writerow(["dimensions"]+x)

            for algo in yValues.iterkeys():
                y = yValues[algo][metric]
                writer.writerow([algo] + y)

def runExperimentForMetric(data, label, algos, dimensions):

    yValues = dict()
    for algo in algos:
        yValues[algo] = dict()
        x = list()
        #y = list()

        yValues[algo]["rocAuc"] = list()
        yValues[algo]["algoDuration"] = list()
        yValues[algo]["lrDuration"] = list()

        for dimension in dimensions:

            x.append(dimension)

            print "%s -> %s dimensions" % (algo, dimension)

            algoDuration, reduced, score = durationRocAuc(algo, data, label, dimension)
            lrDuration = measureFitLR(reduced, label)
            yValues[algo]["rocAuc"].append(score)
            yValues[algo]["algoDuration"].append(algoDuration)
            yValues[algo]["lrDuration"].append(lrDuration)

        #yValues[algo]["rocAuc"] = np.mean(yValues[algo]["rocAuc"])
        #yValues[algo]["algoDuration"] = np.mean(yValues[algo]["algoDuration"])
        #yValues[algo]["lrDuration"] = np.mean(yValues[algo]["lrDuration"])

    return x, yValues

def setupExperimentFolder(experiment):
    outputFolder = os.path.dirname(os.path.abspath(__file__))
    outputFolder = "%s/experiments_new/%s" % (outputFolder, experiment["name"])
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    print
    print "experiment output is: %s" % outputFolder
    return outputFolder

def loadData(experiment):
    if experiment.has_key("size"):
        size = experiment["size"]
    else:
        size = 0
    data, label, description, reduce = experiment["dataset"]()

    if size > 0:
        initialReduceBlockSize = np.arange(size, size+0.2, 0.1)
        testSetPercentage = 0.2
        trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = data_factory.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, testSetPercentage)

        data = trainDataBlocks[0][0]
        label = trainLabelBlocks[0][0]

    if experiment['binary_encode']:
        print "perform binary encode"
        analyze(data, label, "before encode")
        # encode features (one-hot-encoder / dummy coding)
        enc = OneHotEncoder()
        enc.fit(data)
        data = enc.transform(data).toarray()
        analyze(data, label, "after encode")

    return data, label, description, reduce
