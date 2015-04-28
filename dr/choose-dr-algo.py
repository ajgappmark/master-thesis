import dr
import experiment_run as run
import data_factory
import numpy as np

'''
    documentation of this experiments in google docs, results on github
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
    'description':      'cancer dataset, removed a few algos',
    'name':             'experiment1.2',
    'dataset':          data_factory.loadFirstCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgosExlude(["tsne", "mds", "matrix_factorisaton"]),
    'dimensions':       range(4,15),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

experiment13                = experiment12
experiment13["description"] = "same algos, but more dimensions"
experiment13["name"]        = "experiment1.3"
experiment13["dimensions"]  = np.arange(50,250, 10)

finalAlgos = dr.getAllAlgosInclude(["rp", "hash", "pca", "isomap"])
experiment14                = experiment13
experiment14["description"] = "final algos for that dataset"
experiment14["name"]        = "experiment1.4"
experiment14["algos"]       = finalAlgos

#################### second cancer dataset ##########################
'''
experiment21 = {
    'description':      '2. cancer dataset',
    'name':             'experiment2.1',
    'dataset':          data_factory.loadSecondCancerDataset,
    'binary_encode':    True,
    'algos':            dr.getAllAlgos(),
    'dimensions':       range(4,15),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}
'''
#################### plista dataset ##########################
experiment31 = {
    'description':      '6. plista dataset',
    'name':             'experiment3.1',
    'dataset':          data_factory.loadSixthPlistaDataset,
    'size':             0.1,
    'binary_encode':    False,
    'algos':            finalAlgos,
    'dimensions':       range(4,8),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

experiment32 = experiment31
experiment32["name"]        = "experiment3.2"
experiment32["dimensions"]  = range(4,15)
experiment32["algos"]       = dr.getAllAlgosInclude(["rp", "pca", "hash"])

experiment33 = experiment32
experiment33["name"]        = "experiment3.3"
print experiment33
print experiment33['size']

del(experiment33['size'])



run.execute(experiment32)
