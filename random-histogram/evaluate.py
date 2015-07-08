import random

import matplotlib.pyplot as plt
import os.path
import numpy as np

import collections

'''
Author: Sebastian Alfers
This file is part of the master thesis about Dimensionality Reduction
'''

iterations = 1000000

buckets = dict()
for i in range(0,iterations):
    r = random.randrange(0,100)
    # bucket =  "%.1f" % (r%10)
    bucket = float(r)
    # print bucket
    if float(bucket) in buckets:

        buckets[float(bucket)] += 1
    else:
        #print float(bucket)
        buckets[float(bucket)] = 1

buckets = collections.OrderedDict(buckets)

print list(buckets.iterkeys())

plt.bar(list(buckets.iterkeys()), list(buckets.values()), 0.01, color="black")
plt.grid()
plt.title("histogram of python's random.random()")
plt.xlabel("bins")
plt.ylabel("amount of hits / bin")
folder = os.path.dirname(os.path.abspath(__file__))
plt.savefig("%s/output/python-random-hist.png" % folder, dpi=320)
