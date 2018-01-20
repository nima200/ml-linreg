import numpy as np
import matplotlib.pyplot as plt
import util as utl

# Load data
train = np.genfromtxt('Dataset_1_train.csv', delimiter=',', usecols=(0, 1))
valid = np.genfromtxt('Dataset_1_valid.csv', delimiter=',', usecols=(0, 1))
test = np.genfromtxt('Dataset_1_test.csv', delimiter=',', usecols=(0, 1))
# Create
N = len(train)
P = 20
train_x = train[:, 0]
train_y = train[:, 1]
valid_x = valid[:, 0]
valid_y = valid[:, 1]
test_x = test[:, 0]
test_y = test[:, 1]

X, Y = utl.calcXY(train, P)

#  L2 Regularization added
lamb = 0
lambs = np.array([lamb])
W = utl.calcW(X, Y, P, lamb)
mse_train = np.array([utl.calcMSE(W, train_x, train_y, P, N)])
mse_validation = np.array([utl.calcMSE(W, valid_x, valid_y, P, N)])
mse_test = np.array([utl.calcMSE(W, test_x, test_y, P, N)])
while lamb <= 1:
    lamb += 0.001
    lambs = np.append(lambs, lamb)
    w_new = utl.calcW(X, Y, P, lamb)
    mse_train = np.append(mse_train, utl.calcMSE(w_new, train_x, train_y, P, N))
    mse_validation = np.append(mse_validation, utl.calcMSE(w_new, valid_x, valid_y, P, N))
    mse_test = np.append(mse_test, utl.calcMSE(w_new, test_x, test_y, P, N))

mse = np.array([lambs, mse_train, mse_validation, mse_test]).T
min_mse_valid_index = np.where(mse_validation == min(mse_validation))
print(mse[min_mse_valid_index])
bestLambda = mse[min_mse_valid_index, 0][0][0]
minW = utl.calcW(X, Y, P, bestLambda)
xx = np.linspace(min(test_x), max(test_x), N * 100)
yy = np.array([])

for i in range(0, N * 100):
    row = utl.makePoly(xx[i], P)
    yy = np.append(yy, minW.T.dot(row))


plt.figure(1)
plt.plot(xx, yy, color='b')
plt.scatter(test_x, test_y, color='r')
plt.show()

