import data_factory as df
import numpy as np

cancer = df.loadFirstCancerDataset()

print cancer
print np.shape(cancer)