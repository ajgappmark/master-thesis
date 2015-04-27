import numpy as np
import cancer_datasets as cancer


def analyze(data, label, description = "analyze data"):
    shapeData = np.shape(data)
    shapeLabel = np.shape(label)
    if shapeData[0] != shapeLabel[0]:
        raise Exception("each instance needs a label")
    print "------ %s -------" % description
    print np.shape(data)
    print np.shape(label)

    numSamples = len(label)

    positiveExamples = np.sum(label)
    negativeExamples = numSamples - positiveExamples

    negativePercentage = float(negativeExamples) * 100 / float(numSamples)
    positivePercentage = float(positiveExamples) * 100 / float(numSamples)

    print "'0'-labeled data: %s (%.4f %%)" % (negativeExamples, negativePercentage)
    print "'1'-labeled data: %s (%.4f %%)" % (positiveExamples, positivePercentage)
