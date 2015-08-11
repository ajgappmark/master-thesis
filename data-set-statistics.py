'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import data_factory as data
from analyze import analyze

for load in data.getAllDatasets():
    data, label = load()
    analyze(data, label)
