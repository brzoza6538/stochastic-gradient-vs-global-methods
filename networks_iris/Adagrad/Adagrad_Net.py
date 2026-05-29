import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.preprocessing import MinMaxScaler


from Net import *




class AdagradNetwork:
    def __init__(self, layers: List, learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = None

        self.counter = 0
        self.epoch = 0

        self.seen_checkpoints = set()
        self.checkpoints = globals.def_checkpoints
        self.log = {checkpoint: [] for checkpoint in self.checkpoints}

        self.E = 1e-8

    def compile(self, loss):
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calculate_acc_checkpoint(self, x_test, y_test):
        y_pred = self(x_test)


        pred = np.argmax(y_pred, axis=1)
        true = np.argmax(y_test, axis=1)
        return np.mean(pred == true)

    def calculate_loss_checkpoint(self, x_test, y_test):
        y_pred = self(x_test)

        loss_grad = self.loss.calculate_loss(y_test, y_pred)
        return np.mean(loss_grad)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            verbose: int = 0):

        num_samples = x_train.shape[0]
        self.counter = 0
        self.seen_checkpoints = set()

        for layer in self.layers:
            if isinstance(layer, FullyConnected):
                layer.w_cache = np.zeros_like(layer.weights)
                layer.b_cache = np.zeros_like(layer.bias)

        while self.counter < MAX_EVALS:

            permutation = np.random.permutation(num_samples)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)

                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                y_pred = x_batch
                for layer in self.layers:
                    y_pred = layer.forward(y_pred)

                loss_grad = self.loss.loss_derivative(y_batch, y_pred)

                for layer in reversed(self.layers):
                    loss_grad = layer.backward(loss_grad)

                for layer in self.layers:
                    if isinstance(layer, FullyConnected):

                        layer.w_cache += layer.weights_derivative ** 2
                        layer.b_cache += layer.bias_derivative ** 2

                        layer.weights -= (
                            self.learning_rate *
                            layer.weights_derivative /
                            (np.sqrt(layer.w_cache) + self.E)
                        )

                        layer.bias -= (
                            self.learning_rate *
                            layer.bias_derivative /
                            (np.sqrt(layer.b_cache) + self.E)
                        )

                        layer.weights_derivative.fill(0)
                        layer.bias_derivative.fill(0)

                self.counter += len(x_batch)

                for checkpoint in self.checkpoints:
                    checkpoint_fes = int(checkpoint * MAX_EVALS)

                    if checkpoint not in self.seen_checkpoints and self.counter >= checkpoint_fes:
                        loss_grad = self.calculate_loss_checkpoint(x_test, y_test)
                        acc = self.calculate_acc_checkpoint(x_test, y_test)
                        self.log[checkpoint].append(loss_grad)
                        self.seen_checkpoints.add(checkpoint)

                        if verbose:
                            print(f"Checkpoint {checkpoint}: acc={acc}")

                if self.counter >= MAX_EVALS:
                    if verbose:
                        print(f"Reached MAX_EVALS = {MAX_EVALS}")
                    return

            self.epoch += 1
            if verbose:
                print("epoch:", self.epoch)

                

def run_adagrad_net(run_id, images, labels,  seed=None):

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
    tanh_layer3 = Tanh()

    my_network = AdagradNetwork(layers=[
                                fully_connected_layer1,
                                tanh_layer1, fully_connected_layer2,
                                tanh_layer2, fully_connected_layer3,
                                tanh_layer3
                                ], learning_rate=0.05)


    my_loss = Loss(def_loss,def_derivative_loss)

    my_network.compile(loss=my_loss)

    my_network.fit(x_train, y_train, x_test, y_test, verbose=1)

    log = my_network.log
    result = []

    for checkpoint in log.keys():
        result.append({
            "algorithm": 'adagrad',
            "run": run_id,
            "checkpoint": checkpoint,
            "error": log[checkpoint]
            })
        
    print(run_id)

    return result




