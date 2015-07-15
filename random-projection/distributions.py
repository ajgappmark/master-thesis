import random
import numpy as np
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

import build_rp_matrix

def getRand():
    return random.randrange(0,100)

'''
dense density distribution
+1 with 1/2
-1 with 1/2
'''
def dense_1(orig_rows, orig_columns, new_columns):

    def dist(rows, columns):
        rand = getRand()
        if rand < 50:
           return 1
        else:
           return -1

    return build_rp_matrix.buildMatrix(orig_columns, new_columns, dist)


'''
dense density distribution

m refers to the amount of observations

Gaussian with N(0, 1/m)
'''
dense_2_rand = np.random.mtrand._rand
def dense_2(orig_rows, orig_columns, new_columns):

    def dist(rows, columns):
        m = rows
        #return np.random.normal(0.0, (1.0 / float(m)), 1)
        return dense_2_rand.normal(loc=0.0, scale= 1.0 / np.sqrt(new_columns), size = 1)

    return build_rp_matrix.buildMatrix(orig_columns, new_columns, dist)

        # return dense_2_rand.normal(loc=0.0, scale= 1.0 / np.sqrt(new_columns), size = (orig_rows, new_columns))

'''
sparse density distribution
1 with 1/6
-1 with 1/6
0 with 2/3
'''
def sparse_1(orig_rows, orig_columns, new_columns):

    def dist(rows, columns):
        scale = np.sqrt(3)
        bound = (1.0/6.0)*100
        rand = getRand()
        if rand < bound:
           return scale
        elif rand > 100-bound:
            return -scale
        else:
            return 0

    return build_rp_matrix.buildMatrix(orig_columns, new_columns, dist)




'''
sparse density distribution
n refers to the amount of dimension in the original matrix

sqrt(n) with 1/2*sqrt(n)
-sqrt(n) with 1/2*sqrt(n)
0 with 1-(1/sqrt(n))

'''
def sparse_2(orig_rows, orig_columns, new_columns):

    def dist(rows, columns):
        n = columns
        scale = np.sqrt(n)
        bound = (1.0/(2*np.sqrt(n) )) * 100
        rand = getRand()
        if rand > 50 and rand <= 50+bound:
           return scale
        elif rand > 50-bound and rand <= 50:
            return -scale
        else:
            return 0

    return build_rp_matrix.buildMatrix(orig_columns, new_columns, dist)

'''
sparse density distribution
n refers to the amount of dimension in the original matrix

s = 1 / density
density = 1/sqrt(n)

-sqrt(s) / sqrt(n) with 1/2s
0 with 1-1/s
sqrt(s) / sqrt(n) with 1/2s

'''
def sparse_3(orig_rows, orig_columns, new_columns):

    def dist(rows, columns):
        n = columns
        density = 1.0/np.sqrt(n)
        s = 1.0 / density

        ret = np.sqrt(s) / np.sqrt(n)
        bound = 1.0/2.0*n
        rand = getRand()
        if rand < bound:
            return ret
        elif rand > 100-bound:
            return -ret
        else:
            return 0

    return build_rp_matrix.buildMatrix(orig_columns, new_columns, dist)

all = {
    "dense 1": dense_1,
    "sparse 1": sparse_1,
    "sparse 2": sparse_2,
    "sparse 3": sparse_3,
    "dense 2": dense_2,
}

dense2 = {
    "dense 2": dense_2
}

def getAll():
  return all
