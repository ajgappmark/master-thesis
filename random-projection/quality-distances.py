import numpy as np
import data_factory as df
import dimensionality_reduction as dr
from scipy.spatial.distance import euclidean
import build_rp_matrix as rp
import distributions
from sklearn.feature_extraction import FeatureHasher
import time
from scipy.sparse import bsr_matrix
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scikit_rp
from scipy.sparse import csc_matrix, csr_matrix

# accumulate the distance from each row to each other by euclidean dist
def getError(dataA, dataB):
    if(hasattr(dataA, "toarray")):
        dataA = dataA.toarray()
    if(hasattr(dataB, "toarray")):
        dataB = dataB.toarray()
    a = pdist(dataA, "euclidean")
    b = pdist(dataB, "euclidean")
    difference = a - b
    return np.abs(np.mean(difference))

distributions.getAll()
scikit_rp_implementations = {
    "scikit gaussian" : scikit_rp.gaussianRP,
    "scikit sparse":    scikit_rp.sparseRP
}


def runExperiment(setup):

    experimentId = setup["id"]
    customDistributions = setup["customDistributions"]
    scikitImplementations = setup["scikitImplementations"]
    data = setup["data"]
    data, _,_,_ = data()
    orig_shape = np.shape(data)
    print "orig rows / cols: %s/%s " % (orig_shape[0], orig_shape[1])
    dimensions = np.arange(50, 350, 50)
    plt.xlabel("dimensions")
    plt.ylabel("error")
    plt.grid()
    for key in customDistributions:
        dist = customDistributions[key]
        errors = []
        for dimension in dimensions:
            avg_errors = []
            for k in range(0, 20):
                randomMatrix = dist(orig_shape[0], orig_shape[1], dimension)
                if isinstance(randomMatrix, np.ndarray):
                    randomMatrix = csr_matrix(randomMatrix)
                reduced = data * randomMatrix
                error = getError(data, reduced)
                avg_errors.append(error)
            errors.append(np.mean(avg_errors))
        print "%s: %s" % (key, errors)
        plt.plot(dimensions, errors, label="%s (%.2f)" %(key, np.mean(errors)))

    # lets see, how the implementations of scikit-learn perform
    for key in scikitImplementations:
        errors = []
        for dimension in dimensions:

            avg_errors = []
            for k in range(0, 20):

                action = scikitImplementations[key]
                reduced = action(data, int(dimension))
                error = getError(data, reduced)
                avg_errors.append(error)
            errors.append(np.mean(avg_errors))
        print "%s: %s" % (key, np.mean(errors))
        plt.plot(dimensions, errors, label="%s (%.2f)" %(key, np.mean(errors)))

    plt.legend(loc="best")
    plt.savefig("output/experiment_%s_quality-distances.png" % experimentId , dpi=320)


################### experiments ###################
experiment1 = {
    "id":   1,
    "customDistributions": distributions.all,
    "scikitImplementations": scikit_rp_implementations,
    "data": df.loadFirstCancerDataset
}

experiment2 = {
    "id":   2,
    "customDistributions": distributions.dense2,
    "scikitImplementations": scikit_rp_implementations,
    "data": df.loadFirstCancerDataset
}

experiment3 = experiment1.copy()
experiment3["data"] = df.loadSecondCancerDataset
experiment3["id"] = 3

experiment4 = experiment2.copy()
experiment4["data"] = df.loadSecondCancerDataset
experiment4["id"] = 4

def getFirstPlistaData():
    data, label, _, _ = df.loadFirstPlistaDataset()
    initialReduceBlockSize = np.arange(0.01, 0.3, 0.1)
    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = df.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, 0.1)
    return trainDataBlocks[0][0], _,_,_

def getSecondPlistaData():
    data, label, _, _ = df.loadSecondPlistaDataset()
    initialReduceBlockSize = np.arange(0.01, 0.3, 0.1)
    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = df.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, 0.1)
    return trainDataBlocks[0][0], _,_,_

def getThirdPlistaData():
    data, label, _, _ = df.loadThridPlistaDataset()
    initialReduceBlockSize = np.arange(0.01, 0.3, 0.1)
    trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = df.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, 0.1)
    return trainDataBlocks[0][0], _,_,_



experiment5 = experiment1.copy()
experiment5["data"] = getFirstPlistaData
experiment5["id"] = 5

experiment6 = experiment2.copy()
experiment6["id"] = 6
experiment6["data"] = getFirstPlistaData

experiment7 = experiment2.copy()
experiment7["id"] = 7
experiment7["data"] = getSecondPlistaData

experiment8 = experiment2.copy()
experiment8["id"] = 8
experiment8["data"] = getThirdPlistaData


runExperiment(experiment8)