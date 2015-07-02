import os.path
import numpy as np
import urllib2

datasets = [
            ["https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/", "wdbc.data"],
            ["https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/", "breast-cancer-wisconsin.data"]
          ]

downloadFolder = "data/"

def getLocalFolder(file):
    folder = os.path.dirname(os.path.abspath(__file__))
    return "%s/%s" %(folder, file)

def fileExists(file):
    file = getLocalFolder(file)
    return os.path.isfile(file)

def downloadFile(urlFolder, fileName):
    response = urllib2.urlopen(urlFolder+fileName)
    data = response.read()
    fileName = getLocalFolder("")+downloadFolder+fileName

    print fileName
    file = open(fileName, 'w+')
    file.write(data)
    file.close

def fileToVector(file, processRow):
    file = getLocalFolder("")+downloadFolder+file
    d = open(file, 'r')
    lines = [line.rstrip('\r\n') for line in d]
    rows = len(lines)

    # count amount of columns, skip first row
    columns = len(lines[0].split(',')) -2

    data = np.empty(shape=(rows, columns))
    label =np.empty(shape=(rows))

    for i in range(rows):
        item = lines[i]
        row = np.array([n for n in item.split(',')])

        # process row according to dataset
        x, y = processRow(row)
        data[i] = sanitizeVector(x)
        label[i] = y
    return data, label

def processLineForFirstDataset(line):
    '''
        first dataset according to my master thesis: "Brest Cancer Wisconsin (Disgnostic)"

        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        a row looks like this:
        Id,      Label,  Features
        842302,  M,      17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189

        Label B = Benign will be encoded as 0
        Label M = Malignant will be encoded as 1
    '''
    # skip first column
    line = line[1:]

    if line[0] == 'B':
        label = 0
    elif line[0] == 'M':
        label = 1
    else:
        raise Exception('only label B or M allowed. label %s is wrong' % line[0])

    l = line[1:]

    return l, label

def processLineForSecondDataset(line):
    '''
        second dataset according to my master thesis: "Breast Cancer Wisconsin (Original) Data Set

        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

        a row looks like this:
        Id,      9 Attributes,      Label
        1000025, 5,1,1,1,2,1,3,1,1, 2

        Label 2 = Benign will be encoded as 0
        Label 4 = Malignant will be encoded as 1
        "
    '''
    label = line[len(line)-1]
    # skip first and last column
    line = line[1:len(line)-1]

    # encode label
    if label == '2':
        label = 0
    elif label == '4':
        label = 1
    else:
        raise Exception('only label 2 or 4 allowed. label %s is wrong' % line[0])

    #l = list()
    #for item in line:
    #    l.append(int(item*1000000))

    return line, label

def sanitizeVector(row):
    '''
        converts values do double and apply
        my simple strategy to handle missing values ('?')
        - replace them as 0 to make them behave as not existing in binary representation
    '''
    newData = np.empty(shape=(len(row)))
    for i in range(len(row)):
        value = row[i]
        if value == '?':
            value = 0
        newData[i] = value
    return newData

def loadBothDatasets():
    for item in datasets:
        urlFolder = item[0]
        file = item[1]
        localFile = downloadFolder+file
        if not fileExists(localFile):
            downloadFile(urlFolder, file)

    # load both in memory
    firstDatasetData, firstDatasetLabel = fileToVector(datasets[0][1], processLineForFirstDataset)
    secondDatasetData,secondDatasetLabel = fileToVector(datasets[1][1], processLineForSecondDataset)

    return firstDatasetData, firstDatasetLabel, secondDatasetData, secondDatasetLabel

def loadFirst():
    firstDatasetData, firstDatasetLabel, secondDatasetData, secondDatasetLabel = loadBothDatasets()
    return firstDatasetData, firstDatasetLabel

def loadSecond():
    firstDatasetData, firstDatasetLabel, secondDatasetData, secondDatasetLabel = loadBothDatasets()
    return secondDatasetData, secondDatasetLabel


