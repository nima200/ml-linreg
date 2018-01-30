import numpy as np


def calcXY(data_set, degree: int):
    X = np.array([])
    Y = np.array(data_set[:, 1]).reshape(-1, 1)
    for data_point in data_set[:, 0]:
        X = np.append(X, makePoly(data_point, degree))
    X = X.reshape(len(data_set), degree + 1)
    return X, Y


def calcW(x: np.ndarray, y: np.ndarray, degree: int, lamb=0):
    i = np.identity(degree + 1)
    w = np.linalg.inv((x.T.dot(x)) + (lamb * i)).dot((x.T.dot(y)))
    return w


def makePoly(x: float, p: int):
    row = np.array([])
    for i in range(0, p + 1):
        row = np.append(row, x**i)
    return row


def linPredict(row: np.ndarray, coeffs: np.ndarray):
    yHat = coeffs[0]
    for i in range(len(coeffs) - 1):
        yHat += coeffs[i + 1] * row[i]
    return yHat


def calcMSE(coeffs: np.ndarray, data_x: np.ndarray, data_y: np.ndarray,
            degree: int, count=-1):
    if count == -1:
        count = len(data_x)
    mse = 0
    for i in range(0, count):
        row = makePoly(data_x[i], degree)
        result = coeffs.T.dot(row)
        mse += ((result - data_y[i]) ** 2)
    mse /= count
    return mse


def save_splits(data: np.array, splits: int):
    chunkSize = int(round(len(data) / splits))
    np.random.shuffle(data)
    for i in range(splits):
        test_start = chunkSize * i
        test_end = test_start + chunkSize
        if i == splits - 1:
            test_start -= 1
        test = data[test_start:test_end]
        train = np.vstack((data[:test_start], data[test_end:]))
        np.savetxt('./data/CandC-train%d.csv' % i, train, delimiter=',')
        np.savetxt('./data/CandC-test%d.csv' % i, test, delimiter=',')


def fillMissingValues(data: np.array):
    col_mean = np.array([0.0 for i in range(len(data[0]))])

    for i in range(len(col_mean)):
        col = data[:, i]
        temp = np.array([])
        for item in col:
            if not np.isnan(item):
                temp = np.append(temp, item)
        col_mean[i] = np.mean(temp)
        for j in range(len(col)):
            if np.isnan(col[j]):
                col[j] = col_mean[i]
        data[:, i] = col
    np.savetxt('./data/Dataset_completed.csv', data, delimiter=',')
    return data


def load_splits():
    train_0 = np.genfromtxt('./data/CandC-train0.csv', delimiter=',')
    train_1 = np.genfromtxt('./data/CandC-train1.csv', delimiter=',')
    train_2 = np.genfromtxt('./data/CandC-train2.csv', delimiter=',')
    train_3 = np.genfromtxt('./data/CandC-train3.csv', delimiter=',')
    train_4 = np.genfromtxt('./data/CandC-train4.csv', delimiter=',')
    test_0 = np.genfromtxt('./data/CandC-test0.csv', delimiter=',')
    test_1 = np.genfromtxt('./data/CandC-test1.csv', delimiter=',')
    test_2 = np.genfromtxt('./data/CandC-test2.csv', delimiter=',')
    test_3 = np.genfromtxt('./data/CandC-test3.csv', delimiter=',')
    test_4 = np.genfromtxt('./data/CandC-test4.csv', delimiter=',')
    test = np.array([test_0, test_1, test_2, test_3, test_4])
    train = np.array([train_0, train_1, train_2, train_3, train_4])
    return train, test


def inverseSet(data: np.array, cap: int):
    out = np.array([])
    for i in range(cap):
        if not np.isin(i, data):
            out = np.append(out, i)
    return out.astype(int)
