'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import numpy as np

'''
construct a random matrix base on given distribution 'draw'
'''
def buildMatrix(rows, columns, draw):
    rm = np.empty((rows, columns))

    for i in range(len(rm)):
        for j in range(len(rm[i])):
            rm[i][j] = draw(rows, columns)

    return rm