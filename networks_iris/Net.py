import numpy as np
from keras.datasets import mnist

from abc import ABC, abstractmethod

from typing import List
from algorithms import globals

# FULL_MNIST = 784
FULL_IRIS = 4

# INPUT = 200
HID_LAYER_1 = 3
HID_LAYER_2 = 3

# OUTPUT = 10
IRIS_OUTPUT = 3

BATCH_SIZE = 120

MAX_EVALS = 100000

CLAMPS = [-1, 1]





def def_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((pred_y - val_y)**2)

def def_derivative_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((2 * (pred_y - val_y)))


# def cross_entropy_loss(y_true, y_pred):
#     return -np.sum(y_true * np.log(y_pred + 1e-8), axis=1)

# def derivative_cross_entropy(y_true, y_pred):
#     return (y_pred - y_true) / y_true.shape[0]


class Layer(ABC):

    def __init__(self) -> None:
      pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        pass

class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int, weight_range=0.01) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

        self.weights_derivative = np.zeros((input_size, output_size))
        self.bias_derivative = np.zeros((1, output_size))

        self.inputs = None

        self.w_m = np.zeros_like(self.weights)
        self.w_v = np.zeros_like(self.weights)
        self.b_m = np.zeros_like(self.bias)
        self.b_v = np.zeros_like(self.bias)
        self.t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        info = (np.matmul(x, self.weights) + (self.bias * self.input_size))

        return info

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        self.weights_derivative += np.matmul(self.inputs.T, output_error_derivative)
        input_error_derivative = np.matmul(output_error_derivative, self.weights.T)

        self.bias_derivative += np.sum(output_error_derivative, axis=0, keepdims=True)
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
    
class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        return (self.inputs > 0).astype(float) * output_error_derivative

class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        self.outputs = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        input_gradient = np.empty_like(output_error_derivative)

        for i, (y, grad) in enumerate(zip(self.outputs, output_error_derivative)):
            y = y.reshape(-1, 1)
            jacobian = np.diagflat(y) - y @ y.T
            input_gradient[i] = jacobian @ grad

        return input_gradient

    
class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss_function(y_true, y_pred)

    def loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_true, y_pred)

class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = None
        self.counter = 0
        self.seen_checkpoints = set()
        self.checkpoints = globals.def_checkpoints
        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.epoch = 0 

        self.E=1e-8
        self.B1=0.9 
        self.B2=0.999 


    def compile(self, loss: Loss) -> None:
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return(x)

    def calculate_loss_checkpoint(self, x_test, y_test):
        y_pred = self(x_test)

        loss_grad = self.loss.calculate_loss(y_test, y_pred)
        # mean loss over all samples
        return np.mean(loss_grad)


    def calculate_acc_checkpoint(self, x_test, y_test):
        y_pred = self(x_test)  # forward entire test set at once

        pred = np.argmax(y_pred, axis=1)
        true = np.argmax(y_test, axis=1)
        acc = np.mean(pred == true)

        return acc


    # def collect_data(self, x_test, y_test):
    #     print("cntr - err \t", self.counter, " - ", error)
    #     for checkpoint in self.checkpoints:
    #         checkpoint_fes = int(checkpoint * MAX_EVALS)
            
    #         if self.counter <= checkpoint_fes:
    #             error = self.calculate_test_acc(x_test, y_test)

    #             if error < globals.def_smallest_val :
    #                 self.log[checkpoint].append(0)

    #         if checkpoint not in self.seen_checkpoints and self.counter >= checkpoint_fes:
    #             self.log[checkpoint].append(0 if error < globals.def_smallest_val else error)
    #             self.seen_checkpoints.add(checkpoint)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            verbose: int = 0,
            ):



        num_samples = x_train.shape[0]
        self.counter = 0
        self.seen_checkpoints = set()

        while self.counter < MAX_EVALS:
            # Shuffle training data each epoch
            permutation = np.random.permutation(num_samples)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for start_idx in range(0, num_samples, BATCH_SIZE):
                # print("epoch: ", self.epoch, " \t batch : ", start_idx)
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # --- Forward pass ---
                y_pred = x_batch
                for layer in self.layers:
                    y_pred = layer.forward(y_pred)

                # --- Backward pass ---
                loss_derivative = self.loss.loss_derivative(y_batch, y_pred)
                for layer in reversed(self.layers):
                    loss_derivative = layer.backward(loss_derivative)

                # --- Update weights for FullyConnected layers ---
                for layer in self.layers:
                    if isinstance(layer, FullyConnected):
                        # TODO - Adam
                        layer.t += 1

                        # Wagi
                        layer.w_m = self.B1 * layer.w_m + (1 - self.B1) * layer.weights_derivative
                        layer.w_v = self.B2 * layer.w_v + (1 - self.B2) * (layer.weights_derivative ** 2)
                        m_hat = layer.w_m / (1 - self.B1 ** layer.t)
                        v_hat = layer.w_v / (1 - self.B2 ** layer.t)
                        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.E)

                        # Biasy
                        layer.b_m = self.B1 * layer.b_m + (1 - self.B1) * layer.bias_derivative
                        layer.b_v = self.B2 * layer.b_v + (1 - self.B2) * (layer.bias_derivative ** 2)
                        m_hat = layer.b_m / (1 - self.B1 ** layer.t)
                        v_hat = layer.b_v / (1 - self.B2 ** layer.t)
                        layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.E)


                        # Reset derivatives
                        layer.weights_derivative.fill(0)
                        layer.bias_derivative.fill(0)

                # --- Update counters ---
                self.counter += len(x_batch)

                # --- Collect checkpoint data ---
                for checkpoint in self.checkpoints:
                    checkpoint_fes = int(checkpoint * MAX_EVALS)
                    if checkpoint not in self.seen_checkpoints and self.counter >= checkpoint_fes:
                        print(self.counter)
                        acc = self.calculate_acc_checkpoint(x_test, y_test)
                        loss_grad = self.calculate_loss_checkpoint(x_test, y_test)
                        print(acc)
                        self.log[checkpoint].append(loss_grad)
                        self.seen_checkpoints.add(checkpoint)

                # Stop if MAX_EVALS reached
                if self.counter >= MAX_EVALS:
                    if verbose:
                        print(f"Reached MAX_EVALS = {MAX_EVALS}")
                    return
            
            self.epoch += 1
            print("epoch : ", self.epoch)



class EmbedLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.weights = (
            np.random.randn(input_size, output_size)
            / np.sqrt(output_size)
        )
        self.bias = np.zeros((1, output_size))
        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        info = (np.matmul(x, self.weights) + self.bias)

        return info
    
    def backward(self, error):
        return None
    