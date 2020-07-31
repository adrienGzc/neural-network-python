import argparse
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from modules import Layer, utils

def load_dataset(datasetWanted):
    dataset = {
        'iris': datasets.load_iris(),
        'wine': datasets.load_wine(),
        'cancer': datasets.load_breast_cancer()
    }

    if datasetWanted not in dataset:
        raise TypeError('As to be one of: ' + ', '.join(list(dataset.keys())))
    tmp = dataset[datasetWanted]
    return train_test_split(preprocessing.normalize(tmp['data']),
        tmp['target'], test_size=0.30, random_state=42)


def main(params):
    X_train, X_test, Y_train, Y_test = load_dataset(params['dataset'])
    # X_train = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    # Y_train = np.array([[0, 1, 1, 0]])

    # m = 4
    epochs = int(params['epochs'])

    layers = [Layer.Layer(105, 32, 'tanh'),
        Layer.Layer(32, 16, 'sigmoid'),
        Layer.Layer(16, 3, 'sigmoid')
    ]
    # costs = []

    for _epoch in range(epochs):
        # Feedforward
        A = X_train
        for layer in layers:
            A = layer.feedforward(A)

        # Calulate cost to plot graph
        # cost = 1 / m * np.sum(utils.logloss(Y_training, A))
        # costs.append(cost)

        # Backpropagation
        xx = 1.0 - A
        print(xx.shape, A.shape)
        dA = utils.logloss_derivatif(Y_train, A)
        for layer in reversed(layers):
            dA = layer.backprop(dA)

    # Making predictions
    A = X_train
    for layer in layers:
        A = layer.feedforward(A)
    print(A)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=False, help="Dataset wanted.")
    parser.add_argument("-e", "--epochs", required=False, help="Number of epochs.")
    parser.add_argument("-n", "--neurons", required=False, nargs="+", help="Number of neurons in each layer.")
    params = vars(parser.parse_args())

    main(params)

    # nn = NN.NN([120, 30, 3])
    # nn.SGD(train, 30, 10, 3.0, test_data=test)