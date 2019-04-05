import numpy as np


def linear(x):
    return x


def sigmoid(x):
    return np.divide(1, np.add(1, np.exp(np.multiply(-1, x))))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def lrelu(x):
    if x >= 0:
        return x
    else:
        return np.multiply(0.01, x)


def prelu(x, a):
    if x >= 0:
        return x
    else:
        return np.multiply(a, x)


def selu(x, a=1):
    if x >= 0:
        return x
    else:
        return np.multiply(a, np.subtract(np.exp(x), 1))
