'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import matplotlib.pyplot as plt
import csv
import os.path


plt.grid()
plt.title("continuous values")
plt.xlabel("amount of rooms")
plt.ylabel("price of house (in thousand $)")

plt.legend(loc="best")
folder = os.path.dirname(os.path.abspath(__file__))


x = []
y_pca = []
y_rp = []

with open("%s/results.csv" % folder, 'rb') as csvfile:
    stats = csv.reader(csvfile, delimiter=',')
    rows = []
    for row in stats:
        rows.append(row)

    iterRows = iter(rows)
    next(iterRows)

    iterRows = iter(sorted(iterRows, key=lambda row: float(row[0] )))
    next(iterRows)

    for row in iterRows:
        x.append(float(row[0]) + float(row[1]))
        y_pca.append(float(row[2]))
        y_rp.append(float(row[3]))
    print rows


plt.plot(x,y_pca, label='pca')
plt.plot(x,y_rp, label='random projection')

plt.legend(loc="best")


plt.savefig("%s/output/final-evaluation.png" % folder, dpi=320)