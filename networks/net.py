import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from sklearn.datasets import fetch_openml

from algorithms import CMAES


from typing import List
import numpy as np
from abc import ABC, abstractmethod

import numpy as np
from algorithms import globals
from algorithms import CMAVariation, eswrapper, Eval_wrapper 

import time
from functools import partial
import numpy as np

FULL_MNIST = 784

INPUT = 100
HID_LAYER_1 = 16
HID_LAYER_2 = 16

OUTPUT = 10

TEST_BATCH_SIZE = 20


mnist = fetch_openml('mnist_784')
images, labels = mnist.data, mnist.target.astype(int)

clamp = [-1, 1]


def def_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((pred_y - val_y)**2)

def def_derivative_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((2 * (pred_y - val_y))) # sum?
    # TODO - potraktować jak 


class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
      pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int, weight_range=0.01) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(input_size, output_size) * weight_range
        self.bias = np.random.rand(1, output_size) * weight_range

        self.weights_derivative = np.zeros((input_size, output_size))
        self.bias_derivative = np.zeros((1, output_size))

        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        info = (np.matmul(x, self.weights) + self.bias)

        return info

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        self.weights_derivative += np.matmul(self.inputs.T, output_error_derivative)
        input_error_derivative = np.matmul(output_error_derivative, self.weights.T)

        self.bias_derivative += (output_error_derivative)
        return input_error_derivative

class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return np.tanh(x)

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        help = (1 - np.tanh(self.inputs)**2) * output_error_derivative

        return (help)


class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the loss for a particular prediction and true value"""
        return self.loss_function(y_true, y_pred)

    def loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the derivative of the loss function for a particular prediction and true value"""
        return self.loss_function_derivative(y_true, y_pred)

class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = None

    def compile(self, loss: Loss) -> None:
        """Define the loss function and loss function derivative"""
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)

        return(x)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            verbose: int = 0,
            batch_size=1) -> None:
        """Fit the network to the training data"""
        if verbose == 2:
          data = []
        for epoch in range(epochs):
            total_loss = 0
            index = 0

            for x, y_true in zip(x_train, y_train):
                index += 1
                # forward
                x = x.reshape(1,self.layers[0].input_size)
                y_pred = self(x)

                # calculate loss
                current_loss = self.loss.calculate_loss(y_true, y_pred)

                total_loss += np.sum(current_loss)

                # backpropagtion

                loss_derivative = self.loss.loss_derivative(y_true, y_pred)

                for layer in reversed(self.layers):
                    loss_derivative = layer.backward(loss_derivative)

                # Update weights and biases
                if(index % batch_size == 0):
                  for layer in self.layers:
                    if isinstance(layer, FullyConnected):
                      help = layer.weights
                      layer.weights -= (self.learning_rate * layer.weights_derivative) / batch_size
                      layer.bias -= (self.learning_rate * layer.bias_derivative) / batch_size

                      layer.weights_derivative = np.zeros((layer.input_size, layer.output_size))
                      layer.bias_derivative = np.zeros((1, layer.output_size))

            # verbose options
            if verbose == 1:
              print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(x_train)}")
            if verbose == 2:
              data.append(total_loss / len(x_train))
        if verbose == 2:
          return data


class EmbedLayer(Layer):
    def __init__(self, input_size: int, output_size: int, weight_range=0.01) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(input_size, output_size) * weight_range
        self.bias = np.random.rand(1, output_size) * weight_range

        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        info = (np.matmul(x, self.weights) + self.bias)

        return info
    
    def backward(self, error):
        return None
    





######### struct 

class Evaluation_method():

    def __init__(self, seed):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=seed) # TODO = add a  seed that changes every time?

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # Normalize pixel values to be between 0 and 1
        self.x_train = self.x_train / 255.0
        self.x_test =  self.x_test / 255.0

        # Flatten the images
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)


        # Convert class vectors to binary class matrices
        self.lb = LabelBinarizer()
        self.y_train = self.lb.fit_transform(self.y_train)
        self.y_test = self.lb.transform(self.y_test)


        # Enmbedder
        self.E_fully_connected_layer = EmbedLayer(input_size=FULL_MNIST, output_size=INPUT)
        self.tanh_layer0 = Tanh()

        # Instantiate layers
        self.fully_connected_layer1 = FullyConnected(input_size=INPUT, output_size=HID_LAYER_1)
        self.tanh_layer1 = Tanh()
        self.fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
        self.tanh_layer2 = Tanh()
        self.fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=OUTPUT)
        self.tanh_layer3 = Tanh()

        self.my_network = Network(layers=[self.E_fully_connected_layer, self.tanh_layer0,
                                    self.fully_connected_layer1,
                                    self.tanh_layer1, self.fully_connected_layer2,
                                    self.tanh_layer2, self.fully_connected_layer3,
                                    self.tanh_layer3
                                    ], learning_rate=0.01)

        # Compile the network with a loss function

        self.my_loss = Loss(def_loss,def_derivative_loss)

        self.my_network.compile(loss=self.my_loss)
        self.train_pointer = 0


    def evaluate(self, x):
        # Y = self.objective_f.evaluate(x)
        # error = abs(Y - self.global_min)
        # evaluations_used = 1
        # return error, evaluations_used
        pointer = 0

        for layer in self.my_network.layers:
            if isinstance(layer, FullyConnected):
                snippet = x[pointer : pointer + (layer.input_size * layer.output_size)]
                snippet = snippet.reshape(layer.input_size, layer.output_size)
                layer.weights = snippet
                pointer += layer.input_size * layer.output_size

                snippet = x[pointer : pointer + layer.output_size]
                snippet = snippet.reshape(1, layer.output_size)
                layer.bias = snippet
                pointer += layer.output_size

        train_loss = 0
        correct_predictions = 0

        if self.train_pointer * TEST_BATCH_SIZE > len(self.x_train):
            self.train_pointer = 0

        l = self.train_pointer * TEST_BATCH_SIZE

        for x, y_true in zip(self.x_train[ l :  l + TEST_BATCH_SIZE], self.y_train[ l : l + TEST_BATCH_SIZE]):
            #x = x.reshape(1, -1)
            y_pred = self.my_network(x)

            # Calculate loss (you might want to use the proper loss function here)
            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            train_loss += current_loss

            # Check if the prediction is correct
            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / len(self.x_train)
        self.train_pointer += 1
        # Calculate average loss and accuracy
        return (1-accuracy), TEST_BATCH_SIZE
    




##############




def run_cmaes_net(run_id,  seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    dimension = ((INPUT + 1)*HID_LAYER_1 + (HID_LAYER_1 + 1)*HID_LAYER_2 + (HID_LAYER_2 + 1)*OUTPUT)
    x0 = np.random.uniform(clamp[0], clamp[1], size=dimension)
    # switch_interval = 1
    popsize = int(4 + np.floor(3 * np.log(dimension)))

    f_eval = Eval_wrapper(globals.Evaluation_method(seed).evaluate)


    data = eswrapper(
        x=x0,
        fun=f_eval,
        popsize=popsize,
        maxevals=globals.def_max_fes * dimension,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=None,
    )

    result = []

    max_fes = globals.def_max_fes * dimension
    for checkpoint in globals.def_checkpoints:
        eval_checkpoint = max_fes * checkpoint

        idx = np.abs(data.nums_evals - eval_checkpoint).argmin()


        closest_checkpoint = data.nums_evals[idx]

        if( abs(data.nums_evals[idx] - eval_checkpoint ) < 50 ):
            # closest_value = abs(float(curr_f["global_min"]) - data.midpoint_values[idx])
            closest_value = data.best_values[idx]
            result.append({
                "algorithm": 'cmaes',
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value]
            })
        else:
            closest_value = 0
            result.append({
                "algorithm": 'cmaes',
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value]
            })


    return result





def run_adam_net(run_id,  seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=seed)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test =  x_test / 255.0

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)


    # Convert class vectors to binary class matrices
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)


    # Enmbedder
    E_fully_connected_layer = EmbedLayer(input_size=784, output_size=784)
    tanh_layer0 = Tanh()

    # Instantiate layers
    fully_connected_layer1 = FullyConnected(input_size=784, output_size=16)
    tanh_layer1 = Tanh()
    fully_connected_layer2 = FullyConnected(input_size=16, output_size=16)
    tanh_layer2 = Tanh()
    fully_connected_layer3 = FullyConnected(input_size=16, output_size=16)
    tanh_layer3 = Tanh()
    fully_connected_layer4 = FullyConnected(input_size=16, output_size=10)
    tanh_layer4 = Tanh()

    # Instantiate the network
    my_network = Network(layers=[E_fully_connected_layer, tanh_layer0,
                                fully_connected_layer1,
                                tanh_layer1, fully_connected_layer2,
                                tanh_layer2, fully_connected_layer3,
                                tanh_layer3, fully_connected_layer4,
                                tanh_layer4
                                ], learning_rate=0.01)

    # Compile the network with a loss function

    my_loss = Loss(def_loss,def_derivative_loss)

    my_network.compile(loss=my_loss)

    # Train the network

    my_network.fit(x_train, y_train, epochs=5, verbose=1)
