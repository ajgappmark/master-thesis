'''
Author: Sebastian Alfers
This file is part of my thesis 'Evaluation and implementation of cluster-based dimensionality reduction'
License: https://github.com/sebastian-alfers/master-thesis/blob/master/LICENSE
'''

import dr
import experiment_run as run
import data_factory
import numpy as np
import sys

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

experiment13                = experiment12.copy()
experiment13["description"] = "same algos, but more dimensions"
experiment13["name"]        = "experiment1.3"
experiment13["dimensions"]  = np.arange(50,250, 10)

finalAlgos = dr.getAllAlgosInclude(["rp", "hash", "pca", "isomap"])
experiment14                = experiment13.copy()
experiment14["description"] = "final algos for that dataset"
experiment14["name"]        = "experiment1.4"
experiment14["algos"]       = dr.getAllAlgosInclude(["no_DR", "rp", "hash", "pca", "isomap"])
experiment14["dimensions"]  = np.arange(10,70, 10)


#################### second cancer dataset ##########################

experiment21 = experiment11.copy()
experiment21["description"] = "cancer dataset 2"
experiment21["name"]        = "experiment2.1"
experiment21["dataset"]     = data_factory.loadSecondCancerDataset

experiment22 = experiment21.copy()
experiment22["name"]        = "experiment2.2"
experiment22["algos"]       = dr.getAllAlgosExlude(["mds"])

experiment23 = experiment22.copy()
experiment23["name"]        = "experiment2.3"
experiment23["algos"]       = dr.getAllAlgosExlude(["mds", "tsne"])
experiment23["dimensions"]  = np.arange(5,65, 5)

experiment24 = experiment23.copy()
experiment24["name"]        = "experiment2.4"
experiment24["algos"]       = dr.getAllAlgosExlude(["mds", "tsne", "matrix_factorisaton"])

experiment25 = experiment24.copy()
experiment25["name"]        = "experiment2.5"
experiment25["algos"]       = dr.getAllAlgosInclude(["no_DR", "rp", "hash", "incremental_pca", "pca"])
experiment25["dimensions"]  = np.arange(5, 15)

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
del(experiment33['size'])

experiment34 = experiment33
experiment34["name"]        = "experiment3.4"
experiment34["dimensions"]  = np.arange(50, 250, 10)

experiment35 = experiment34
experiment35["dataset"]     = data_factory.loadFirstPlistaDataset
experiment35["name"]        = "experiment3.5"
experiment35["size"]        = 0.4
# experiment35["algos"]       = dr.getAllAlgosInclude(["pca"])
experiment35["dimensions"]  = [20, 30, 50, 100, 150, 200, 250] #np.arange(50, 250, 10)

experiment36 = {
    'description':      '6. plista dataset',
    'name':             'experiment3.1',
    'dataset':          data_factory.loadSecondPlistaDataset,
    'size':             0.01,
    'binary_encode':    False,
    'algos':            dr.getAllAlgosInclude(["tsne"]),
    'dimensions':       [3,4,5, 10, 15, 20],
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

#################### plista dataset ##########################
experiment41 = {
    'description':      '6. plista dataset',
    'name':             'experiment4.1',
    'dataset':          data_factory.loadSixthPlistaDataset,
    'size':             0.1,
    'binary_encode':    False,
    'algos':            dr.getAllAlgosExlude(["tsne", "lle"]),
    'dimensions':       range(4,8),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

experiment42 = experiment41.copy()
experiment42["algos"] = dr.getAllAlgosExlude(["tsne", "lle", "kernel_pca", "spectralEmbedding", "mds", "isomap"])
experiment42["name"]  = "experiment4.2"

experiment43 = experiment42.copy()
experiment43["name"]  = "experiment4.3"
experiment43["dimensions"] = np.arange(10,35, 5)

experiment44 = experiment43.copy()
experiment44["name"]  = "experiment4.4"
experiment44["algos"] = dr.getAllAlgosInclude(["rp", "srp", "hash", "incremental_pca", "pca", "truncated_svd"])
experiment44["dimensions"] = np.arange(10,100, 10)

experiment45 = experiment44.copy()
experiment45["name"]  = "experiment4.5"
experiment45["size"] = 0.2

experiment46 = experiment45.copy()
experiment46["name"]  = "experiment4.6"
experiment46["algos"] = dr.getAllAlgosInclude(["no_DR", "rp", "srp", "hash", "truncated_svd"])
experiment46["size"] = 0.3
experiment46["dimensions"] = np.arange(20,350, 50)

experiment47 = experiment46.copy()
experiment47["name"]  = "experiment4.7"
experiment47["size"] = 0.3

experiment48 = experiment47.copy()
experiment48["name"]  = "experiment4.8"
experiment48["size"] = 0.4

experiment49 = experiment48.copy()
experiment49["name"]  = "experiment4.9"
experiment49["dataset"] = data_factory.loadFirstPlistaDataset

experiment410 = experiment48.copy()
experiment410["name"]  = "experiment4.10"
experiment410["size"] = 0.5
experiment410["dimensions"] = np.arange(5,30, 5)

experiment411 = experiment48.copy()
experiment411["name"]  = "experiment4.11"
experiment411["size"] = 0.6
experiment411["dimensions"] = np.arange(5,10)

experiment412 = experiment411.copy()
experiment412["name"]  = "experiment4.12"
experiment412["size"] = 0.6
experiment412["algos"] = dr.getAllAlgosInclude(["no_DR", "hash"])

#################### plista dataset ##########################

# MEMORY ERROR TSNE -> 4 dimensions

all = {
    "11": experiment11,
    "12": experiment12,
    "13": experiment13,
    "14": experiment14,

    "21": experiment21,
    "22": experiment22,
    "23": experiment23,
    "24": experiment24,
    "25": experiment25,

    "31": experiment31,
    "32": experiment32,
    "33": experiment33,
    "34": experiment34,
    "35": experiment35,
    "36": experiment36,

    "41": experiment41,
    "42": experiment42,
    "43": experiment43,
    "44": experiment44,
    "45": experiment45,
    "46": experiment46,
    "47": experiment47,
    "48": experiment48,
    "49": experiment49,
    "410": experiment410,
    "411": experiment411,
    "412": experiment412,


}
experiment51 = {
    'description':      '2. plista dataset',
    'name':             'experiment4.1',
    'dataset':          data_factory.loadSecondPlistaDataset,
    'size':             0.05,
    'binary_encode':    False,
    'algos':            dr.getAllAlgos(),
    'dimensions':       range(4,11),
    'yValues':          ['rocAuc', 'algoDuration', 'lrDuration']
}

if len(sys.argv) != 2:
    print "only / max one param allowed"
    exit()

params = sys.argv
id = params[1]
if not all.has_key(id):
    print "experiment with id '%s' not found" % id
    exit()

run.execute(all[id])
