from scipy.sparse import csr_matrix
import numpy as np

def processLine(line, lineIndex):
    lineItems = line.split(',')
    # first item is the label
    label = float(lineItems[0])

    # prune the label
    lineItems = lineItems[1:]

    # collect data
    row = list()
    col = list()
    data = list()

    for value in lineItems:
        colVal = value.split(':')
        row.append(int(lineIndex))      # i'th value represents row index
        col.append(int(colVal[0]))      # i'th value represents column index
        data.append(float(colVal[1]))   # i'th value represents value

    return label, row, col, data

def convert(fileName):
    file = open(fileName, 'r')
    lines = [line.rstrip('\r\n') for line in file]
    amountRows = len(lines)

    rows = list()
    cols = list()
    data = list()
    labels = list()

    for lineIndex in range(len(lines)):

        if lineIndex % 50000 == 0:
            print "%s lines processed..." % lineIndex

        line = lines[lineIndex]

        label, r, c, d = processLine(line, lineIndex)
        labels.append(int(label))
        rows.extend(r)
        cols.extend(c)
        data.extend(d)

    print "all %s lines processed" % amountRows

    maxRows = np.max(rows)
    maxColumns = np.max(cols)

    matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(maxRows+1, maxColumns+1))
    return matrix, labels

def getStatistics(labels, matrix):
    shape = np.shape(matrix)
    instances = shape[0]
    features = shape[1]

    positive = np.sum(labels)
    negative = instances - positive

    return instances, features, positive, negative

'''
print "dataset 1"
matrix, labels = convert("../plista-data/export_sparse_publisher_vectors_11_1477_hour_3-1.csv")
print getStatistics(labels, matrix)

print "dataset 2"
matrix, labels = convert("../plista-data/export_sparse_publisher_vectors_11_13725,970,4787_hour_2-1.csv")
getStatistics(labels, matrix)
'''
