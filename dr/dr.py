from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, IncrementalPCA, NMF, TruncatedSVD, KernelPCA
import time
from sklearn import random_projection
import numpy as np

def hash(data, labels, new_dimension):
    print "start hashing trick..."
    start = time.time()
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


def pca(data, labels, new_dimension):
    print "start pca..."
    start = time.time()
    pca = PCA(n_components=new_dimension)

    if hasattr(data, "toarray"):
        data = data.toarray()

    reduced = pca.fit_transform(data)
    end = time.time()
    return (reduced, end-start)

def ipca(data, labels, new_dimension):
    print "start incremental pca..."
    start = time.time()
    pca = IncrementalPCA(n_components=new_dimension, batch_size=1000)

    if hasattr(data, "todense"):
        reduced = pca.fit_transform(np.array(data.todense()))
    else:
        reduced = pca.fit_transform(data)

    end = time.time()
    return (reduced, end-start)

def kernelPCA(data, labels, new_dimension):
    print "start kernel pca..."
    start = time.time()
    pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=new_dimension)
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

def reduceByKey(key, d, l, dimensionValue):
    options = {
        'hash': hash,
        'rp': randomProjection,
        'pca': pca,
        'incremental_pca': ipca,
        'kernel_pca': kernelPCA,
        'truncated_svd': truncatedSVD,
        'non_negative_matrix_factorisaton': nnMatrixFactorisation
    }
    return options[key](d,l, dimensionValue)