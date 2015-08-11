'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

print(__doc__)


from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll

import dr
import data_factory as df
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


###############################################################################
# Generate data (swiss roll dataset)
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise)
# Make it thinner
X[:, 1] *= .5

X = X + np.abs(np.min(X[:,0]))
X = X + np.abs(np.min(X[:,1]))
X = X + np.abs(np.min(X[:,2]))


###############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")

ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
label = ward.labels_
print("Number of points: %i" % label.size)

###############################################################################
# Plot result
colors = list()


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10., azim=100)

#for l in np.unique(label):
for l in label:
    colors.append( plt.cm.jet(float(l) / np.max(label + 1)) )
    # ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
    #          'o', color=plt.cm.jet(float(l) / np.max(label + 1)))

for lbl, data in zip(colors, X):
    ax.scatter(xs = data[0], ys = data[1], zs = data[2], c=lbl)
plt.title('No DR')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.legend(loc="best")
plt.savefig("output/siwss_3d.png", dpi=320)


algos = dr.getAllAlgos()

for desc in algos:
    run = algos[desc]
    reduced, durations = run(X, colors, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)


    for lbl, data in zip(colors, reduced):
        ax.scatter(x = data[0], y = data[1], c=lbl)
    plt.title('algorithm: %s' % desc)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend(loc="best")
    plt.savefig("output/siwss_2d_%s.png" % desc, dpi=320)
