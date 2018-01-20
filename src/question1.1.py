import numpy as np
import matplotlib.pyplot as plt
import util as utl

# Load data
train = np.genfromtxt('Dataset_1_train.csv', delimiter=',', usecols=(0, 1))
valid = np.genfromtxt('Dataset_1_valid.csv', delimiter=',', usecols=(0, 1))
# Create
N = len(train)
P = 20
train_input = train[:, 0]
train_target = train[:, 1]
valid_input = valid[:, 0]
valid_target = valid[:, 1]

X = np.array([])
for x in train_input:
    X = np.append(X, utl.makePoly(x, P))
X = X.reshape(N, P + 1)

Y = train_target.reshape(-1, 1)

minW = np.linalg.inv((X.T.dot(X))).dot((X.T.dot(Y)))

xx = np.linspace(min(train_input), max(train_input), N * 100)
yy = np.array([])

for i in range(0, N * 100):
    row = utl.makePoly(xx[i], P)
    yy = np.append(yy, minW.T.dot(row))

mse_train = 0.0
# Mean Squared Error calculation - Train set
for i in range(0, N):
    row = utl.makePoly(train_input[i], P)
    y_model = minW.T.dot(row)
    mse_train += (y_model - train_target[i]) ** 2
mse_train /= N
print(mse_train)

mse_valid = 0.0
# Mean Squared Error calculation - Validation set
for i in range(0, N):
    row = utl.makePoly(valid_input[i], P)
    y_model = minW.T.dot(row)
    mse_valid += (y_model - valid_target[i]) ** 2
mse_valid /= N
print(mse_valid)

plt.figure(1)
plt.plot(xx, yy, color='b')
plt.scatter(train_input, train_target, color='r')
plt.title('Linear Model generated by training data - Degree 20')
plt.show()
