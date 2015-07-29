'''
this file reads output from spark and plots the results
'''

import matplotlib.pyplot as plt
import csv
import numpy as np

with open('/home/sebastian_alfers/results_pairwise_spark.csv', 'rb') as csvfile:
    stats = csv.reader(csvfile, delimiter=',')

    rows = []
    for row in stats:
        rows.append(row)
    len = np.shape(rows)[0]
    x = []
    y = []
    for item in rows[1 : len-2]:
        x.append(item[0])
        y.append(item[1])

    mean = rows[len-1][1]

    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.grid()

    plt.plot(x, y, label="error of pairwise distances")

    plt.legend(loc="best")
    plt.savefig("output/pairwise/pairwise_spark.png", dpi=320)

    print "mean error %s" % mean