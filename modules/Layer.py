import numpy as np
import importlib

from modules import utils

class Layer:
    def __init__(self, inputs, neurons, activation):
        self.__inputs = inputs
        self.__neurons = neurons
        self.__activation, self.__activation_derivatif = self.__getActivationFunction(activation)
        self.__weights = np.random.randn(neurons, inputs)
        self.__bias = np.zeros((neurons, 1))
        self.__learning_rate = 0.1

    def __getActivationFunction(self, activation_function):
        possible_function = {
            'sigmoid': (utils.sigmoid, utils.sigmoid_derivatif),
            'tanh': (utils.tanh, utils.tanh_derivatif)
        }
        return possible_function.get(activation_function)

    def feedforward(self, A_prev):
        self.A_prev = A_prev
        print(self.__weights.shape, self.A_prev.shape)
        self.Z = np.dot(self.__weights, self.A_prev) + self.__bias
        print('Z: ', self.Z.shape)
        self.A = self.__activation(self.Z)
        return self.A
    
    def backprop(self, dA):
        dZ = np.multiply(self.__activation_derivatif(self.Z), dA)
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.__weights.T, dZ)

        self.__weights = self.__weights - self.__learning_rate * dW
        self.__bias = self.__bias - self.__learning_rate * db

        return dA_prev