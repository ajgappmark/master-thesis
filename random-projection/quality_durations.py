'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import distributions
import scikit_rp
import data_factory as df
import time


datasets = {}
datasets["cancer_1"] = df.loadFirstCancerDataset
datasets["cancer_2"] = df.loadSecondCancerDataset
datasets["plista_1"] = df.loadFirstPlistaDataset
datasets["plista_2"] = df.loadSecondPlistaDataset
datasets["plista_3"] = df.loadThridPlistaDataset
datasets["plista_4"] = df.loadFourthPlistaDataset
i = 0
for dataset in datasets.iterkeys():
    import matplotlib.pyplot as plt

    plt.figure(i)
    i = i+1
    plt.xlabel('dimensions')
    plt.ylabel('durations in seconds')
    plt.title('dataset: %s' % dataset)

    data,_,_,_ = datasets[dataset]()
    data = csr_matrix(data)

    orig_columns = np.shape(data)[1]

    dimensions = np.arange(50, 350, 50)
    durations = {}
    durations["dense 2"] = []
    durations["scikit gaussian"] = []
    durations["scikit sparse"] = []

    for dimension in dimensions:

        rand_matrices = {}

        matrixDense1 = csc_matrix(np.transpose(np.transpose(distributions.dense_1(0, orig_columns=orig_columns, new_columns=dimension))))

        matrixSparse1 = csc_matrix(np.transpose(np.transpose(distributions.sparse_1(0, orig_columns=orig_columns, new_columns=dimension))))
        matrixSparse2 = csc_matrix(np.transpose(np.transpose(distributions.sparse_2(0, orig_columns=orig_columns, new_columns=dimension))))
        matrixSparse3 = csc_matrix(np.transpose(np.transpose(distributions.sparse_3(0, orig_columns=orig_columns, new_columns=dimension))))

        matrixDense2 = csc_matrix(np.transpose(np.transpose(distributions.dense_2(0, orig_columns=orig_columns, new_columns=dimension))))
        matrixB = csc_matrix(np.transpose(scikit_rp.getGaussianRP(dimension)._make_random_matrix(dimension, orig_columns)))
        matrixC = csc_matrix(np.transpose(scikit_rp.getSparseRP(dimension)._make_random_matrix(dimension, orig_columns).toarray()))

        rand_matrices["dense 1"] = matrixDense1
        rand_matrices["sparse 1"] = matrixSparse1
        rand_matrices["sparse 2"] = matrixSparse2
        rand_matrices["sparse 3"] = matrixSparse3
        rand_matrices["dense 2"] = matrixDense2

        rand_matrices["scikit gaussian"] = matrixB
        rand_matrices["scikit sparse"] = matrixC

        for key in rand_matrices.iterkeys():
            matrix = rand_matrices[key]

            avg_durations = []
            for i in range(0, 20):
                start = time.time()
                reduced = data * matrix
                duration = time.time() - start
                avg_durations.append(duration)
            durations[key].append(np.mean(avg_durations))

    for key in rand_matrices.iterkeys():
        plt.plot(dimensions, durations[key], label=key )
    plt.legend(loc="best")

    plt.savefig("output/durations/dimensions_vs_durations_%s.png" % dataset, dpi=320)
    plt.close()