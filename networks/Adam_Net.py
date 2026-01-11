import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import time

from Net import *


def run_adam_net(func, run_id,  seed=None):

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
    E_fully_connected_layer = EmbedLayer(input_size=FULL_MNIST, output_size=INPUT)
    tanh_layer0 = Tanh()

    # Instantiate layers
    fully_connected_layer1 = FullyConnected(input_size=INPUT, output_size=HID_LAYER_1)
    tanh_layer1 = Tanh()
    fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
    tanh_layer2 = Tanh()
    fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=OUTPUT)
    tanh_layer3 = Tanh()

    # Instantiate the network
    my_network = Network(layers=[E_fully_connected_layer, tanh_layer0,
                                fully_connected_layer1,
                                tanh_layer1, fully_connected_layer2,
                                tanh_layer2, fully_connected_layer3,
                                tanh_layer3
                                ], learning_rate=0.01)

    # Compile the network with a loss function

    my_loss = Loss(def_loss,def_derivative_loss)

    my_network.compile(loss=my_loss)

    # Train the network

    my_network.fit(x_train, y_train, x_test, y_test, verbose=1)

    log = my_network.log
    result = []

    for checkpoint in log.keys():
        result.append({
            "function": func,
            "run": run_id,
            "checkpoint": checkpoint,
            "error": log[checkpoint]
            })
    return result
