import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import os.path

regr = linear_model.LinearRegression()

# amount of rooms per observation
X = [[1] ,[2] ,[3] ,[3] ,[5] ,[6]]
# price per observation
Y = [[20],[30],[35],[38],[50],[55]]

plt.figure(figsize=(10,5))
plt.subplot(121)
regr.fit(X, Y)
plt.grid()
plt.title("continuous values")
plt.xlabel("amount of rooms")
plt.ylabel("price of house ($)")

plt.scatter(X, Y, label='data points',  color='black')
plt.plot(X, regr.predict(X), label='approximated values', color='blue', linewidth=3)

plt.subplot(122)

# map the category to (integer) value to be able to map in on an axis
appartement = 10
bungalow    = 20
flat        = 30
cottage     = 40
villa       = 50

# category of observation
X = [[appartement] ,[appartement] ,[bungalow] ,[bungalow] ,[flat] ,[flat] ,[villa] ,[villa]]
# price per observation
Y = [[20],[25],[44],[47],[52],[59],[70],[90]]

regr_category = linear_model.LinearRegression()

regr_category.fit(X, Y)

plt.grid()
plt.title("continuous values")
plt.xlabel("category of house")

# Plot outputs
plt.scatter(X, Y, label='data points',  color='black')
plt.plot(X, regr_category.predict(X), label='approximated values', color='blue', linewidth=3)


plt.legend(loc="best")

folder = os.path.dirname(os.path.abspath(__file__))

plt.savefig("%s/output/categorical-vs-continuous.png" % folder, dpi=320)

# print the output into a file for referencing in thesis
with open('log.txt', 'w') as file:
    file.write('##### predict the price for houses (rooms) #####\n')
    rooms = 4
    file.write('rooms: %s\n' % rooms)
    prediction = regr.predict(rooms)
    file.write('prediction: %s\n' % prediction)

    file.write('\n')

    file.write('##### predict the price for houses (house category) #####\n')
    file.write("ID of category 'cottage': %s\n" % cottage)
    prediction = regr_category.predict(cottage)
    file.write('prediction: %s\n' % prediction)

    file.write('\n')

    cottage = 60
    file.write("changed ID of category 'cottage': %s\n" % cottage)
    prediction = regr_category.predict(cottage)

    file.write('prediction: %s\n' % prediction)