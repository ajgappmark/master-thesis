from scipy.sparse import *
from scipy import *
import numpy as np
import time
from sklearn import linear_model
import data_factory as df
import matplotlib.pyplot as plt
import dr

plt.figure()
plt.xlabel('dimensions')
plt.ylabel('duration (seconds)')
plt.title('ROC Curves')

data, label, desc, size = df.loadFirstPlistaDataset()

initialReduceBlockSize = np.arange(0.1, 0.7, 0.1)
trainDataBlocks, trainLabelBlocks, testDataBlocks, testLabelBlocks = df.splitDatasetInBlocks(data, np.array(label), initialReduceBlockSize, 0.1)
data = trainDataBlocks[0][0]
label = trainLabelBlocks[0][0]

x = list()
sparse_y = list()
dense_y = list()

algos = dr.getAllAlgosExlude(["tsne"])

algodurations = {}

dimension_range = np.arange(50, 250, 50)
for i in dimension_range: # dimensions

    x.append(i)

def measureLR(data, label):
    start = time.time()
    lr = linear_model.LogisticRegression()
    lr.fit(data, label)
    end = time.time()
    return end - start

for desc in algos:
    algodurations[desc] = {}
    algodurations[desc]['sparse'] = list()
    algodurations[desc]['dense'] = list()

    run = algos[desc]
    for i in dimension_range:
        reduced, _ = run(data, label, i)
        lrDurations = measureLR(reduced, label)
        algodurations[desc]['sparse'].append(lrDurations)

for desc in algodurations:
    d = algodurations[desc]
    plt.plot(x, d['sparse'], label=desc)
    # plt.plot(x, dense_y, label="dense")

plt.legend(loc="best")

plt.savefig("output/compare.png", dpi=320)
