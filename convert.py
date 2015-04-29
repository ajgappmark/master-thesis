from scipy.sparse import csr_matrix
import numpy as np
import os

def processLine(line, lineIndex):
    lineItems = line.split(',')
    # first item is the label
    label = float(lineItems[0])

    # collect data
    # prune the label
    lineItems = lineItems[1:]

    row = list()
    col = list()
    data = list()

    for value in lineItems:
        colVal = value.split(':')

        row.append(float(lineIndex))      # i'th value represents row index
        col.append(float(colVal[0]))      # i'th value represents column index
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
    skipped = 0
    for lineIndex in range(len(lines)):

        if lineIndex % 50000 == 0:
            print "%s lines processed..." % lineIndex

        line = lines[lineIndex]

        if line == "0.0," or line == "1.0,":
            #print "skip row..."
            skipped = skipped + 1
            continue

        label, r, c, d = processLine(line, lineIndex - skipped) # important to reduce the index!!!
        labels.append(float(label))
        rows.extend(r)
        cols.extend(c)
        data.extend(d)

    print "all %s lines processed" % amountRows

    maxRows = np.max(rows)
    maxColumns = np.max(cols)


    matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(maxRows+1, maxColumns+1))
    print np.shape(matrix)
    print np.shape(labels)
    print skipped

    return matrix, labels

folder = os.path.dirname(os.path.abspath(__file__))
folder = "%s/data" % folder
def loadFirst():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_33158,970,13725_hour_2-1.csv" % folder)
    return data, labels

def loadSecond():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_33158,970,13725_hour_3-1.csv" % folder)
    return data, labels

def loadThird():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_33158,970,13725_hour_4-1.csv" % folder)
    return data, labels

def loadFirth():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_33158,970,13725,970,4787_hour_4-1.csv" % folder)
    return data, labels

def loadFifth():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_13725,5335_hour_2-1.csv" % folder)
    return data, labels


def loadSixth():
    data, labels = convert("%s/export_sparse_publisher_vectors_11_970_hour_2-1.csv" % folder)
    return data, labels