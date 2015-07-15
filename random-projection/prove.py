import distributions

'''
Author: Sebastian Alfers
This file is part of the master thesis about Dimensionality Reduction
'''



distributions = distributions.getAll()
with open('log.txt', 'w') as file:
    # prove that each distribution behaves as desired
    for key in distributions.iterkeys():
        draw = distributions[key]
        results = dict()
        # draw for 1000 times
        for i in range(0, 100000):
            result = draw()
            if result in results:
                results[result] += 1
            else:
                results[result] = 1

        file.write("###### '%s' ######' \n"  % key)
        for res_key in results.iterkeys():
            file.write("'%s' : %.2f%% \n" % (res_key, results[res_key] / 1000.0))

        file.write("\n")