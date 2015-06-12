from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, IncrementalPCA, NMF, TruncatedSVD, KernelPCA
import time
from sklearn import random_projection, manifold
import numpy as np
from copy import copy

def hash(data, labels, new_dimension):
    print "start hashing trick..."
    # convert features as dict
    dictList = list()
    if hasattr(data, "indices"):
        #ind = data.indices
        #dat = data.data
        data = data.toarray()
        indices = range(len(data[0]))
        for item in data:
            zipped = zip(indices, item)
            row = dict()
            for index,value in zipped:
                if value != 0:
                    row[str(index)] = value
            dictList.append(row)

        a = 234
    else:
        indices = map(str, range(len(data[0])))
        for row in data:
            dictList.append(dict(zip(indices, row)))

    start = time.time()
    hasher = FeatureHasher(n_features=new_dimension) # , input_type='dict'
    reduced = hasher.fit_transform(dictList).toarray()
    end = time.time()
    return (reduced, end-start)

def randomProjection(data, labels, new_dimension):
    print ("start random projection...")
    start = time.time()
    transformer = random_projection.GaussianRandomProjection(n_components=new_dimension)
    reduced = transformer.fit_transform(data)
    end = time.time()
    #print (" took %f" % (end - start))
    return (reduced, end-start)

def sparseRandomProjection(data, label, new_dimension):
    print ("start sparse random projection...")
    start = time.time()
    transformer = random_projection.SparseRandomProjection(n_components=new_dimension)
    reduced = transformer.fit_transform(data)
    end = time.time()
    #print (" took %f" % (end - start))
    return (reduced, end-start)

def pca(data, labels, new_dimension):
    print "start pca..."

    if hasattr(data, "toarray"):
        data = data.toarray()

    start = time.time()
    pca = PCA(n_components=new_dimension)

    reduced = pca.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def ipca(data, labels, new_dimension):
    print "start incremental pca..."

    if hasattr(data, "todense"):
        data = np.array(data.todense())

    start = time.time()
    pca = IncrementalPCA(n_components=new_dimension)
    reduced = pca.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def kernelPCA(data, labels, new_dimension):
    print "start kernel pca..."

    if hasattr(data, "toarray"):
        data = data.toarray()

    start = time.time()
    pca = KernelPCA(fit_inverse_transform=True, gamma=10, n_components=new_dimension, alpha=2)

    reduced = pca.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def truncatedSVD(data, labels, new_dimension):
    print "start truncatedSVD..."
    start = time.time()
    pca = TruncatedSVD(n_components=new_dimension)
    reduced = pca.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def nnMatrixFactorisation(data, labels, new_dimension):
    print "non negative matrix factorisation..."
    start = time.time()
    mf = NMF(n_components=new_dimension)
    reduced = mf.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def tsne(data, labels, new_dimension):
    print "tsne..."

    #if hasattr(data, "toarray"):
    #    data = data.toarray()

    start = time.time()
    tsne = manifold.TSNE(n_components=new_dimension, learning_rate=500)
    reduced = tsne.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def isomap(data, labels, new_dimension):
    print "isomap..."

    if hasattr(data, "toarray"):
        data = data.toarray()

    start = time.time()
    iso = manifold.Isomap(n_components=new_dimension)
    reduced = iso.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def mds(data, labels, new_dimension):
    print "mds ..."

    if hasattr(data, "toarray"):
        data = data.toarray()

    start = time.time()
    mds = manifold.MDS(n_components=new_dimension)
    reduced = mds.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def lle(data, labels, new_dimension):
    print "lle..."

    if hasattr(data, "toarray"):
        data = data.toarray()

    start = time.time()
    lle = manifold.LocallyLinearEmbedding(n_components=new_dimension) # n_neighbors= int(new_dimension/2),
    reduced = lle.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def spectralEmbedding(data, labels, new_dimension):
    print "spectralEmbedding..."

    start = time.time()
    mds = manifold.SpectralEmbedding(n_components=new_dimension)
    reduced = mds.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

options = {
    'hash': hash,
    'rp': randomProjection,
    'srp': sparseRandomProjection,
    'pca': pca,
    'incremental_pca': ipca,
    'kernel_pca': kernelPCA,
    'truncated_svd': truncatedSVD,
    'matrix_factorisaton': nnMatrixFactorisation,
    'tsne': tsne,
    'isomap': isomap,
    'mds': mds,
    'lle': lle,
    'spectralEmbedding': spectralEmbedding
}

def getFewAlgos():
    options = {
        'hash': hash,
        'rp': randomProjection,
        'lle': lle
    }
    return options


def getAllFastAlgos():
    fastOptions = getAllAlgos()
    del(fastOptions["matrix_factorisaton"])
    del(fastOptions["tsne"])
    del(fastOptions["mds"])
    return fastOptions

def getFasterAlgos():
    fasterOptions = getAllAlgos()
    del(fasterOptions["isomap"])
    del(fasterOptions["lle"])
    del(fasterOptions["kernel_pca"])
    return fasterOptions

def getAllAlgos():
    return copy(options)

def getAllAlgosInclude(include):
    allAlgos = getAllAlgos()
    includedAlgos = dict()
    for key in include:
        includedAlgos[key] = allAlgos[key]

    return includedAlgos


def getAllAlgosExlude(exclude):
    allAlgos = getAllAlgos()
    for item in exclude:
        del(allAlgos[item])
    return allAlgos

def reduceByKey(key, d, l, dimensionValue):
    return options[key](d,l, dimensionValue)