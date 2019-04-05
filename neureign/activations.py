import numpy as np


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def lrelu(x):
    if (x >= 0):
        return x
    else:
        return x * 0.01


def prelu(x, a):
    if (x >= 0):
        return x
    else:
        return x * a


def selu(x, a=1):
    if (x >= 0):
        return x
    else:
        return a * (np.exp(x) - 1)
