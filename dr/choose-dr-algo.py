import dr
import experiment_run as run
import data_factory

'''
    Setup experiments to reason about what DR algorithm to chose to implement in a cluster

    experiment 1
    - for each algo, plot dimension (4, 15) against
        - roc auc score
        - duration to build dr

    - questions:
        - what are slow / fast algorithms?
        - what algorithms obtain good scores
'''
experiment1 = {
    'description':      'experiment 1, cancer dataset',
    'name':             'experiment1',
    'dataset':          data_factory.loadFirstCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgos(),
    'dimensions':       range(4,15),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}


experiment21 = {
    'description':      'experiment 2.1, cancer dataset 2',
    'name':             'experiment2.1',
    'dataset':          data_factory.loadSecondCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgos(),
    'dimensions':       range(2,10),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}


run.execute(experiment21)
