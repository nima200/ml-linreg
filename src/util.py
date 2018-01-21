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
