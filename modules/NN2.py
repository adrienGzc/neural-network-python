from modules import Layer

class NN:
    def __init__(self, loss, epochs):
        self.__loss = loss
        self.__epochs = epochs
        self.__layers = []

    def add(self, layer):
        if isinstance(layer, Layer):
            self.__layers.append(layer)
        else:
            print('Error: should be a Layer class.')
            exit(84)
    
    def fit(self):
        pass

    def predict(self):
        pass