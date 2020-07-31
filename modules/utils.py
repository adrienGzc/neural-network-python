import numpy as np

# Activation
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivatif(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivatif(x):
    return 1.0 - np.square(np.tanh(x))


# Loss
def logloss(y, a):
    return -(y * np.log(a) + (1.0 - y) * np.log(1.0 - a))

def logloss_derivatif(y, a):
    return (a - y) / (np.matmul(a, (1.0 - a)))