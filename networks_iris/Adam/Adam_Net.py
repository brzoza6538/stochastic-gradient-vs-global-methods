import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.preprocessing import MinMaxScaler


from Net import *


def run_adam_net(run_id, images, labels,  seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    print("---------seeed-----------")
    print(seed)
    print("---------seeed-----------")

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=seed)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)


    fully_connected_layer1 = FullyConnected(input_size=FULL_IRIS, output_size=HID_LAYER_1)
    tanh_layer1 = Tanh()
    fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
    tanh_layer2 = Tanh()
    fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=IRIS_OUTPUT)
    tanh_layer3 = Softmax()

    my_network = Network(layers=[
                                fully_connected_layer1,
                                tanh_layer1, fully_connected_layer2,
                                tanh_layer2, fully_connected_layer3,
                                tanh_layer3
                                ], learning_rate=0.005)


    my_loss = Loss(cross_entropy_loss,cross_entropy_derivative)

    my_network.compile(loss=my_loss)

    my_network.fit(x_train, y_train, x_test, y_test, verbose=1)

    log = my_network.log
    result = []

    for checkpoint in log.keys():
        result.append({
            "algorithm": 'adam',
            "run": run_id,
            "checkpoint": checkpoint,
            "error": log[checkpoint]
            })
        
    print(run_id)

    return result




