'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import matplotlib.pyplot as plt
import csv
import numpy as np
import os.path

folder = os.path.dirname(os.path.abspath(__file__))
folder = "%s/csv" % folder

def loadFromCSV(file):
    with open(file, 'rb') as csvfile:
        stats = csv.reader(csvfile, delimiter=',')

        rows = []
        for row in stats:
            rows.append(row)
        len = np.shape(rows)[0]
        x = []
        y = []
        for item in rows[1 : len-2]:
            x.append(item[0])
            y.append(float(item[3]))

        return (x,y,np.mean(y))

dataset = [10, 50, 100]

for size in dataset:
    plt.figure(size)
    x_spark,y_spark,mean_spark = loadFromCSV("%s/results_pairwise_spark_%s.csv" % (folder, size))
    x_py,y_py, mean_py = loadFromCSV("%s/result_python_%s.csv" % (folder, size))
    plt.xlabel("dimension")
    plt.ylabel("error")
    plt.grid()

    plt.plot(x_spark, y_spark, label="spark (mean: %.2f)" % mean_spark)
    plt.plot(x_py, y_py, label="python (mean: %.2f)" % mean_py)

    plt.legend(loc="best")
    plt.savefig("output/pairwise/final_compare_python_spark_%s.png" % size, dpi=320)
