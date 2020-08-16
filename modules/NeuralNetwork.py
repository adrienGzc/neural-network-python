import numpy as np
import random

from modules.utils import map_activation

class NeuralNetwork():
    def __init__(self, params):
        # Stock all params given.
        self.__params = params
        # Init the global variable of the class with the params.
        self.__input = int(params['input'])
        self.__hidden_layers = params['hidden_layer']
        self.__output_layer = int(params['output_layer'])
        self.__epochs = int(params['epochs'])
        self.__activation, self.__derivatif = map_activation[params['activation']]
        self.__learning_rate = params['learning_rate']
        # Init weight as a matrix with random value and bias.
        self.__weight_hidden_layers = self.init_hidden_layers()
        self.__bias_hidden_layers = [[-1] * int(neurons) for neurons in self.__hidden_layers]
        self.__weight_hidden = np.random.randn(self.__input, int(self.__hidden_layers[0]))
        print('OLD: ', self.__weight_hidden.shape)
        # self.__bias_hidden = np.array([-1] * self.__hidden_layers)
        # self.__weight_output = np.random.randn(self.__hidden_layers, self.__output_layer)
        # self.__bias_output = np.array([-1] * self.__output_layer)

    def init_hidden_layers(self):
        hidden_layers = np.random.randn(self.__input, int(self.__hidden_layers[0]))
        print(hidden_layers, hidden_layers.shape)
        if len(self.__hidden_layers) == 1:
            print('ONE', np.random.randn(int(self.__hidden_layers[0]), self.__output_layer).shape)
            np.append(hidden_layers, np.random.randn(int(self.__hidden_layers[0]), self.__output_layer), axis=0)
            print('SHAPE: ', hidden_layers.shape)
        else:
            print('MULTIPLE')
            for index, neurons in enumerate(self.__hidden_layers[1:]):
                # print(index, neurons, len(self.__hidden_layers[1:]))
                if index == (len(self.__hidden_layers[1:]) - 1):
                    print('END')
                    np.append(hidden_layers, np.random.randn(int(neurons), self.__output_layer), axis=0)
                else:
                    print('CONTINUE')
                    np.append(hidden_layers, np.random.randn(int(neurons), int(self.__hidden_layers[index])), axis=0)
        print('COUCOU: ', hidden_layers.shape)
        return hidden_layers

    def Backpropagation_Algorithm(self, x):
        DELTA_output = []
        'Stage 1 - Error: OutputLayer'
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1)*(ERROR_output) * self.__derivatif(self.OUTPUT_L2))
        
        # arrayStore = []
        'Stage 2 - Update weights OutputLayer and HiddenLayer'
        for i in range(self.__hidden_layers):
            for j in range(self.__output_layer):
                self.__weight_output[i][j] -= (self.__learning_rate * (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.__bias_output[j] -= (self.__learning_rate * DELTA_output[j])
      
        'Stage 3 - Error: HiddenLayer'
        delta_hidden = np.matmul(self.__weight_output, DELTA_output)* self.__derivatif(self.OUTPUT_L1)
 
        'Stage 4 - Update weights HiddenLayer and InputLayer(x)'
        for i in range(self.__output_layer):
            for j in range(self.__hidden_layers):
                self.__weight_hidden[i][j] -= (self.__learning_rate * (delta_hidden[j] * x[i]))
                self.__bias_hidden[j] -= (self.__learning_rate * delta_hidden[j])

    def fit(self, data, target):  
        total_error = 0
        n = len(data)
        # epoch_array = []
        # error_array = []
        for epoch in range(self.__epochs):
            for idx, inputs in enumerate(data):
                self.output = np.zeros(self.__output_layer)
                'Stage 1 - (Forward Propagation)'
                self.OUTPUT_L1 = self.__activation((np.matmul(inputs, self.__weight_hidden) + self.__bias_hidden))
                self.OUTPUT_L2 = self.__activation((np.matmul(self.OUTPUT_L1, self.__weight_output) + self.__bias_output))
                'Stage 2 - One-Hot-Encoding'
                if(target[idx] == 0):
                    self.output = np.array([1,0,0]) #Class1 {1,0,0}
                elif(target[idx] == 1):
                    self.output = np.array([0,1,0]) #Class2 {0,1,0}
                elif(target[idx] == 2):
                    self.output = np.array([0,0,1]) #Class3 {0,0,1}

                square_error = 0
                for i in range(self.__output_layer):
                    erro = (self.output[i] - self.OUTPUT_L2[i]) ** 2
                    square_error = (square_error + (0.05 * erro))
                    total_error = total_error + square_error

                'Backpropagation : Update Weights'
                self.Backpropagation_Algorithm(inputs)

            total_error = (total_error / n)
            if((epoch % 50 == 0) or (epoch == self.__epochs - 1)):
                print("Epoch ", epoch, "- Total Error: ", total_error)
                # error_array.append(total_error)
                # epoch_array.append(epoch)
        return self

    def predict(self, data, target):
        'Returns the predictions for every element of data'
        my_predictions = []
        'Forward Propagation'
        forward = np.matmul(data,self.__weight_hidden) + self.__bias_hidden
        forward = np.matmul(forward, self.__weight_output) + self.__bias_output
                                 
        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])

        # print(" Number of Sample  | Class |  Output |  Hoped Output  ")
        # for i in range(len(my_predictions)):
        #     if(my_predictions[i] == 0): 
        #         print("id:{}    | Iris-Setosa  |  Output: {}  ".format(i, my_predictions[i], target[i]))
        #     elif(my_predictions[i] == 1): 
        #         print("id:{}    | Iris-Versicolour    |  Output: {}  ".format(i, my_predictions[i], target[i]))
        #     elif(my_predictions[i] == 2): 
        #         print("id:{}    | Iris-Iris-Virginica   |  Output: {}  ".format(i, my_predictions[i], target[i]))
        return my_predictions