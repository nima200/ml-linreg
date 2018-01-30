import numpy as np
from util import save_splits, load_splits, calcW, fillMissingValues, inverseSet
import matplotlib.pyplot as plt

data = np.genfromtxt('./data/Dataset_3.csv', delimiter=',')
data = np.delete(data, [0, 1, 2, 3, 4], 1)
data = fillMissingValues(data)

# UNCOMMENT THIS LINE TO SAVE THE 80-20 SPLITS
save_splits(data, 5)

# Put the split data into respective arrays
train, test = load_splits()
train_x = np.empty(shape=(len(train), len(train[0]), len(train[0][0]) - 1))
train_y = np.empty(shape=(len(train), len(train[0]), 1))
test_x = np.empty(shape=(len(test), len(test[0]), len(test[0][0]) - 1))
test_y = np.empty(shape=(len(test), len(test[0]), 1))

for i in range(len(train)):
    train_x[i] = np.delete(train[i], len(train[i][0]) - 1, axis=1)
    train_y[i] = np.array(train[i][:, len(train[i][0]) - 1]).reshape(-1, 1)
for i in range(len(test)):
    test_x[i] = np.delete(test[i], len(test[i][0]) - 1, axis=1)
    test_y[i] = np.array(test[i][:, len(test[i][0]) - 1]).reshape(-1, 1)

# Add a column of 1's for the bias
new_train_x = np.empty(shape=(len(train_x), len(train_x[0]), len(train_x[0][0]) + 1))
new_test_x = np.empty(shape=(len(test_x), len(test_x[0]), len(test_x[0][0]) + 1))
MSE_test_noreg = np.array([0.0 for i in range(5)])
S = len(train_x)
N = len(train_x[0])
P = len(train_x[0][0])
allW = np.empty(shape=(5, 123, 1))
for i in range(S):
    ones_train = np.array([1.0 for k in range(len(train_x[i]))])
    ones_test = np.array([1.0 for k in range(len(test_x[i]))])
    new_train_x[i] = np.insert(train_x[i], 0, ones_train, axis=1)
    new_test_x[i] = np.insert(test_x[i], 0, ones_test, axis=1)
    W = calcW(new_train_x[i], train_y[i], P)
    allW[i] = W
    MSE_test_noreg[i] = ((test_y[i] - new_test_x[i].dot(W)).T.dot(test_y[i] - new_test_x[i].dot(W))[0][0]) \
                        / len(new_test_x[i])

mean_MSE_test_noreg = np.mean(MSE_test_noreg)
print(mean_MSE_test_noreg)

MSE_reg = np.empty(shape=(2000, 7))
# Add L2 Regularization
for i in range(S):
    lamb = 0.0
    P = len(train_x[i][0])
    for j in range(2000):
        lamb += 0.01
        W_new = calcW(new_train_x[i], train_y[i], P, lamb)
        MSE_reg[j][0] = lamb
        MSE_reg[j][i + 1] = ((test_y[i] - new_test_x[i].dot(W_new)).T.dot(test_y[i] - new_test_x[i].dot(W_new))[0][0]) \
                            / len(new_test_x[i])
for i in range(2000):
    MSE_reg[i][6] = np.mean(MSE_reg[i][1:6])
bestRow = MSE_reg[np.where(MSE_reg[:, 6] == min(MSE_reg[:, 6]))][0]
bestLambda = bestRow[0]
bestMean = bestRow[6]

xx = np.linspace(min(MSE_reg[:, 0]), max(MSE_reg[:, 0]), len(MSE_reg))
yy = MSE_reg[:, 6]

plt.figure(0)
plt.plot(xx, yy, color='r')
plt.plot(bestLambda, bestMean, 'o', color='b')
plt.title('Test MSE with Ridge Regularization')
plt.suptitle('Best Lambda: %.3f, Best MSE %.5f' % (bestLambda, bestMean))
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()

minW = np.empty(shape=(S, P + 1, 1))
selectionThreshold = 0.0
bestFeatures = np.array([])
for i in range(S):
    minW[i] = calcW(new_train_x[i], train_y[i], P, bestLambda)

for i in range(10):
    selectionThreshold += 0.01
    trialFeatures = np.array([])
    MSE_reg_reduced = np.empty(shape=(2000, 7))
    for j in range(S):
        trialFeatures = np.append(trialFeatures, np.where(minW[j] > selectionThreshold)[0])
        trialFeatures = np.unique(trialFeatures).astype(int)
        P = len(trialFeatures) - 1
        deleteFeatures = inverseSet(trialFeatures, len(new_train_x[j][0]))
        lamb = 0.0
        reduced_train_x = np.delete(new_train_x[j], deleteFeatures, 1)
        reduced_test_x = np.delete(new_test_x[j], deleteFeatures, 1)
        for k in range(2000):
            lamb += 0.01
            W_reduced = calcW(reduced_train_x, train_y[j], P, lamb)
            MSE_reg_reduced[k][0] = lamb
            MSE_reg_reduced[k][j + 1] = ((test_y[j] - reduced_test_x.dot(W_reduced)).T
                                         .dot(test_y[j] - reduced_test_x.dot(W_reduced))) / len(reduced_test_x)
    for j in range(2000):
        MSE_reg_reduced[j][6] = np.mean(MSE_reg_reduced[j][1:6])
    trialRow_reduced = MSE_reg_reduced[np.where(MSE_reg_reduced[:, 6] == min(MSE_reg_reduced[:, 6]))][0]
    trialLambda_reduced = trialRow_reduced[0]
    trialMean = trialRow_reduced[6]

    xx = np.linspace(min(MSE_reg_reduced[:, 0]), max(MSE_reg_reduced[:, 0]), len(MSE_reg_reduced))
    yy = MSE_reg_reduced[:, 6]

    plt.figure(i + 1)  # Since figure 1 was and i starts from 0
    plt.plot(xx, yy, color='r')
    plt.plot(trialLambda_reduced, trialMean, 'o', color='b')
    plt.title('Test MSE with Ridge Regularization and Feature Selection')
    plt.suptitle('Best Lambda: %.3f, Best Test MSE: %.5f, Feature count: %d' % (trialLambda_reduced, trialMean, P))
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.show()