import numpy as np
import cancer_datasets as cancer


def analyze(data, labels):
    if len(data) != len(labels):
        raise Exception("each instance needs a label")
    print "------ analyze data -------"
    print np.shape(data)
    print np.shape(labels)

    numSamples = len(labels)

    positiveExamples = np.sum(labels)
    negativeExamples = numSamples - positiveExamples

    negativePercentage = negativeExamples * 100 / numSamples
    positivePercentage = positiveExamples * 100 / numSamples

    print "'0'-labeled data: %s (%.2f %%)" % (negativeExamples, negativePercentage)
    print "'1'-labeled data: %s (%.2f %%)" % (positiveExamples, positivePercentage)

'''
data, labels = cancer.loadFirstDataset()
analyze(data,labels)

data, labels = cancer.loadSecondDataset()
analyze(data,labels)
'''
