import dr
import experiment_run as run
import data_factory
import numpy as np

'''
    Setup experiments to reason about what DR algorithm to chose to implement in a cluster

    ###################################### Experiment #####################################################
    experiment 1.1 (commit b9295f616a44bc2781a8d7e3d87333f8a9ef731f)
    - for each algo, plot dimension (4, 14) against
        - roc auc score
        - duration to build the LR model
        - duration to reduce the data

    - questions:
        - what are slow / fast algorithms in terms of reducing?
        - what algorithms obtain good scores

    - observations:
        - 2 algos (mds + matrix factorisation) become slower with growing dimensions
        - tsne gets faster with growing dimension, but is still the one of the slowest
        - 9 out of 12 have a mean runtime (and constant) runtime around of below 1 second, independent of the dimension

    - conclusion / next steps:
        - run the same tests with a different setup of dimensions (50 to 500) to see if the results are stable
        - we exclude mds and matrix factorisation because of their poor performance
        - we also exclude tsne because it is slow and has a bad score
'''
experiment11 = {
    'description':      'experiment 1.1, cancer dataset',
    'name':             'experiment1.1',
    'dataset':          data_factory.loadFirstCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgos(),
    'dimensions':       range(4,15),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

experiment12 = {
    'description':      'cancer dataset with lots of dimensions',
    'name':             'experiment1.2',
    'dataset':          data_factory.loadFirstCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgosExlude(["tsne", "mds", "matrix_factorisaton"]),
    'dimensions':       range(4,15), #np.arange(50,550, 50),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

run.execute(experiment12)
