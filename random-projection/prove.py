import distributions

'''
Author: Sebastian Alfers
This file is part of the master thesis about Dimensionality Reduction
'''


distributions = {
    "dense 1": distributions.dense_1,
    "sparse 1": distributions.sparse_1,
    "sparse 2": distributions.sparse_2,
    "sparse 3": distributions.sparse_3,
    "dense 2": distributions.dense_2,
    "sparse 4": distributions.sparse_4,
}

with open('log.txt', 'w') as file:
    # prove that each distribution behaves as desired
    for key in distributions.iterkeys():
        draw = distributions[key]
        results = dict()
        # draw for 1000 times
        for i in range(0, 10000):
            result = draw()
            if result in results:
                results[result] += 1
            else:
                results[result] = 1

        file.write("###### '%s' ######' \n"  % key)
        for res_key in results.iterkeys():
            file.write("'%s' : %.2f%% \n" % (res_key, results[res_key] / 100.0))

        file.write("\n")