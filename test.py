import argparse
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from modules.NeuralNetwork import NeuralNetwork

def load_dataset(datasetWanted):
    dataset = {
        'iris': datasets.load_iris(),
        'wine': datasets.load_wine(),
        'cancer': datasets.load_breast_cancer(),
        'digits': datasets.load_digits()
    }

    if datasetWanted not in dataset:
        raise TypeError('As to be one of: ' + ', '.join(list(dataset.keys())))
    tmp = dataset[datasetWanted]
    return train_test_split(preprocessing.normalize(tmp['data']),
        tmp['target'],
        test_size=0.30,
        random_state=42)

def main(params):
    # Load dataset.
    X_train, X_test, Y_train, Y_test = load_dataset(params['dataset'])

    # Create params as dictionnay for simplicity.
    params = {
        'input': int(params['input']),
        'hidden_layer': params['hidden'],
        'output_layer': int(params['output']),
        'epochs': int(params['epochs']),
        'learning_rate': float(params['learning']),
        'activation': params['activation']
    }

    # Init the neural network class.
    NN = NeuralNetwork(params)
    # Start training the neural network.
    NN.fit(X_train, Y_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset wanted.")
    parser.add_argument("-e", "--epochs", required=True, help="Number of epochs.")
    parser.add_argument("-i", "--input", required=True, help="Number of neurons in the first layer.")
    parser.add_argument("-s", "--hidden", required=True, help="Number of neurons in the hidden layers.", nargs='+')
    parser.add_argument("-o", "--output", required=True, help="Number of neurons in the output layer.")
    parser.add_argument("-l", "--learning", required=True, help="Learning rate ratio.")
    parser.add_argument("-a", "--activation", required=True, help="Activation function.", choices=['sigmoid', 'relu', 'tanh'])
    params = vars(parser.parse_args())

    main(params)