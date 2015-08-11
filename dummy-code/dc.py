'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import data_factory as df
import numpy as np

cancer = df.loadFirstCancerDataset()

print cancer
print np.shape(cancer)