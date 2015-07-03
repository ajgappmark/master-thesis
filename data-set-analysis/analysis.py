import numpy as np
import data_factory as df
import os.path
import analyze

sets = df.getAllDatasets()

with open('log.txt', 'w') as file:
    file.write('##### printing the size of each dataset #####\n')
    for load in sets:
        data, label, desc, _ = load()
        shape = np.shape(data)
        file.write("dataset '%s':\n" % desc)
        file.write("rows:%s, dimensions:%s\n" % (shape[0], shape[1]))

        negativeExamples, negativePercentage, positiveExamples, positivePercentage = analyze.analyze(data, label, desc)
        file.write("negative observations: %s (%.2f %%) \n" % (negativeExamples, negativePercentage))
        file.write("positive observations: %s (%.2f %%) \n" % (positiveExamples, positivePercentage))
        file.write("\n")