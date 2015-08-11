'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import random

import matplotlib.pyplot as plt
import os.path
import numpy as np

import collections

iterations = 1000000

#output dict
buckets = dict()
for i in range(0,iterations):
    r = random.randrange(0,100)
    # get bucket
    bucket = float(r)
    # init or update bucket
    if float(bucket) in buckets:
        buckets[float(bucket)] += 1
    else:
        buckets[float(bucket)] = 1

# order
buckets = collections.OrderedDict(buckets)

print list(buckets.iterkeys())

plt.bar(list(buckets.iterkeys()), list(buckets.values()), 0.01, color="black")
plt.grid()
plt.title("histogram of python's random.randrange()")
plt.xlabel("bins")
plt.ylabel("amount of hits / bin")
folder = os.path.dirname(os.path.abspath(__file__))
plt.savefig("%s/output/python-random-hist.png" % folder, dpi=320)
