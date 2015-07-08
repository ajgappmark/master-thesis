import random
import numpy as np

def getRand():
    return random.randrange(0,100)

'''
dense density distribution
+1 with 1/2
-1 with 1/2
'''
def dense_1():
    rand = getRand()
    if rand < 50:
       return 1
    else:
       return -1

'''
dense density distribution

m refers to the amount of observations

0 with 1/2
1/m with 1/2
'''
def dense_2(m = 50):
    rand = getRand()
    if rand < 50:
       return 0
    else:
       return 1.0/m


'''
sparse density distribution
1 with 1/6
-1 with 1/6
0 with 2/3
'''
def sparse_1():
    scale = np.sqrt(3)
    bound = (1.0/6.0)*100
    rand = getRand()
    if rand < bound:
       return scale
    elif rand > 100-bound:
        return -scale
    else:
        return 0


'''
sparse density distribution
n refers to the amount of dimension in the original matrix

sqrt(n) with 1/2*sqrt(n)
-sqrt(n) with 1/2*sqrt(n)
0 with 1-(1/sqrt(n))

'''
def sparse_2(n = 50):
    scale = np.sqrt(n)
    bound = (1.0/(2*np.sqrt(50) )) * 100
    rand = getRand()
    if rand > 50 and rand <= 50+bound:
       return scale
    elif rand > 50-bound and rand <= 50:
        return -scale
    else:
        return 0


'''
sparse density distribution
n refers to the amount of dimension in the original matrix

s = 1 / density
density = 1/sqrt(n)

-sqrt(s) / sqrt(n) with 1/2s
0 with 1-1/s
sqrt(s) / sqrt(n) with 1/2s

'''
def sparse_3(n = 50):

    density = 1.0/np.sqrt(n)
    s = 1.0 / density

    ret = np.sqrt(s) / np.sqrt(n)
    bound = (1.0/(2*n)) * 100
    rand = getRand()
    if rand < bound:
       return ret
    elif rand > 100-bound:
        return -ret
    else:
        return 0