import numpy as np
import matplotlib.pyplot as plt
import util as utl

train = np.genfromtxt('./Dataset_2_train.csv', delimiter=',', usecols=(0, 1))
valid = np.genfromtxt('./Dataset_2_valid.csv', delimiter=',', usecols=(0, 1))
test = np.genfromtxt('./Dataset_2_test.csv', delimiter=',', usecols=(0, 1))

train_x = train[:, 0]
train_y = train[:, 1]
valid_x = valid[:, 0]
valid_y = valid[:, 1]
test_x = test[:, 0]
test_y = test[:, 1]


def sgd(data, l_rate, n_epoch):
    coeffs = np.array([3.0, 3.0])
    mse_train_arr = np.array([])
    mse_valid_arr = np.array([])
    mse_test_arr = np.array([])
    for epoch in range(n_epoch):
        for xy in data:
            yHat = utl.linPredict(xy, coeffs)
            error = yHat - xy[-1]  # assume last element is y
            coeffs[0] = coeffs[0] - l_rate * error
            coeffs[1] = coeffs[1] - l_rate * error * xy[0]
        mse_train_arr = np.append(mse_train_arr, utl.calcMSE(coeffs, train_x, train_y, 1))
        mse_valid_arr = np.append(mse_valid_arr, utl.calcMSE(coeffs, valid_x, valid_y, 1))
        mse_test_arr = np.append(mse_valid_arr, utl.calcMSE(coeffs, test_x, test_y, 1))
        if epoch in range(0, 6) and l_rate == 1*10**-2:
            xs = np.linspace(0, max(train_x), len(train_x) * 100)
            yy = np.array([])
            for x in xs:
                yy = np.append(yy, utl.linPredict(np.array([x]), coeffs))
            plt.scatter(test_x, test_y, color='r')
            plt.plot(xs, yy, color='blue')
            plt.title('Linear Regression against Test data - Sequential Gradient Descent')
            plt.suptitle('Learning Rate: %.6f, Epoch: %d' % (l_rate, epoch))
            plt.show()
    return mse_train_arr, mse_valid_arr, mse_test_arr


for i in range(0, 7):
    rate = 1 * (10 ** -i)
    nEpoch = 1000
    mse_train, mse_valid, mse_test = sgd(train, rate, nEpoch)
    print('Min MSE Train: %.5f, Rate: %.6f' % (min(mse_train), rate))
    print('Min MSE Valid: %.5f, Rate: %.6f' % (min(mse_valid), rate))
    print('Min MSE Test: %.5f, Rate: %.6f' % (min(mse_test), rate))

    xx = np.linspace(0, nEpoch, nEpoch)
    plt.figure(1)
    mse_t, = plt.plot(xx, mse_train, color='r', label='Training set')
    mse_v, = plt.plot(xx, mse_valid, color='blue', label='Validation set')
    plt.legend([mse_t, mse_v], ['Training Set', 'Validation Set'])
    plt.title('Training and validation MSE - Sequential Gradient Descent')
    plt.suptitle('Learning Rate: %.6f' % rate)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show()


