import numpy as np

def buildMatrix(rows, columns, draw):
    rm = np.empty((rows, columns))

    for i in range(len(rm)):
        for j in range(len(rm[i])):
            rm[i][j] = draw(rows, columns)

    return rm