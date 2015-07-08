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

    zero_entries = list()
    non_zero_entries = list()
    for item in data:
        if hasattr(item, "toarray"):
            item = item.toarray()[0]

        non_zero = np.nonzero(item)[0].size
        non_zero_entries.append(non_zero)
        zero_entries.append(item.size - non_zero)


    return negativeExamples, negativePercentage, positiveExamples, positivePercentage, np.mean(zero_entries), np.mean(non_zero_entries)