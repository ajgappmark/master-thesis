print(__doc__)

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
import data_factory as df
import dr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import euclidean_distances


tests = list([df.loadFirstPlistaDataset, df.loadFourthPlistaDataset])

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')

for test in tests:
    data, label, desc, size = test()
    label = np.array(label)

    # data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.33, random_state=42)
    result = cross_validation.StratifiedShuffleSplit(label, 1, test_size=0.3)

    for train_index, test_index in result:
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = label[train_index], label[test_index]


        lr = linear_model.LogisticRegression(class_weight={0.0:0.01, 1.0:0.99})
        lr.fit(data_train, label_train)

        label_predicted = lr.predict(data_test)
        predictions_test = lr.predict_proba(data_test)
        real_predictions = []
        for probability, label in zip(predictions_test, label_test):
            real_predictions.append( probability[1] )

        fpr, tpr, thresholds = roc_curve(label_test, real_predictions)
        roc_auc = auc(fpr, tpr)

        # lets calculate the threshold
        min = 100
        for x,y in zip(fpr, tpr):
            dist = euclidean_distances([x,y], [0,1])
            if dist < min:
                min = dist


        plt.plot(fpr, tpr, label="%s (AUC = %0.2f, threshold = %0.2f)" % (desc, roc_auc, min))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        # plt.show()

plt.savefig("output/roc_auc_automatic.png", dpi=320)


print "---"
print thresholds

