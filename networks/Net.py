import numpy as np
from keras.datasets import mnist

from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml

from typing import List
from algorithms import globals

FULL_MNIST = 784

INPUT = 200
HID_LAYER_1 = 16
HID_LAYER_2 = 16

OUTPUT = 10

TEST_BATCH_SIZE = 20

MAX_FES = 100000

CLAMPS = [-1, 1]




mnist = fetch_openml('mnist_784')
images, labels = mnist.data, mnist.target.astype(int)


def def_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((pred_y - val_y)**2)

def def_derivative_loss(val_y : np.ndarray, pred_y : np.ndarray) -> np.ndarray:
    return((2 * (pred_y - val_y)))


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

    def compile(self, loss: Loss) -> None:
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return(x)


    def calculate_test_loss(self, x_test, y_test):
        total_loss = 0
        index = 0

        for x, y_true in zip(x_test, y_test):
            index += 1
            # forward
            x = x.reshape(1,self.layers[0].input_size)
            y_pred = self(x)

            # calculate loss
            current_loss = self.loss.calculate_loss(y_true, y_pred)

            total_loss += np.sum(current_loss)

        return 1 - (total_loss / len(x_test))

    def collect_data(self, x_test, y_test):
        # print(self.counter, " - ", error)
        error = self.calculate_test_loss(x_test, y_test)
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * MAX_FES)
            
            if error < globals.def_smallest_val and self.counter <= checkpoint_fes:
                self.log[checkpoint].append(0)

            if checkpoint not in self.seen_checkpoints and self.counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if error < globals.def_smallest_val else error)
                self.seen_checkpoints.add(checkpoint)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,

            verbose: int = 0,
            batch_size=1) -> None:
        if verbose == 2:
          data = []
        while True:
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
                
                self.counter += 1
                
                self.collect_data(x_test, y_test)
                
                if self.counter > MAX_FES:
                    return 




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






# run_adam_net(0)
# # 10000
# # {0.01: [0.9983523725250187], 0.1: [0.9837359272621663], 0.2: [0.9675134280735028], 0.3: [0.9512861136771121], 0.4: [0.9350183709614902], 0.5: [0.9188025227730621], 0.6: [0.9025928801237727], 0.7: [0.886384038928273], 0.8: [0.8702088255351574], 0.9: [0.8539829180930221], 1.0: [0.8377531784969092]}

# print("\n\n\n\n")
# print(run_cmaes_net(0))

# # [{'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.01, 'error': [1.0]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.1, 'error': [0.9999821428571428]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.2, 'error': [0.9999821428571428]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.3, 'error': [0.9999464285714286]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.4, 'error': [0.9999107142857143]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.5, 'error': [0.9999464285714286]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.6, 'error': [1.0]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.7, 'error': [1.0]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.8, 'error': [0.9999285714285714]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 0.9, 'error': [1.0]}, {'algorithm': 'cmaes', 'dimension': 2058, 'run': 0, 'checkpoint': 1.0, 'error': [0.9999107142857143]}]




# #max_fes = 100 000, EMbed = 200
# # BASE_DIR = /home/plgrid/plgmichalbrz/take_1
# # {0.01: [0.9837694452997668], 0.1: [0.8430897365359662], 0.2: [0.6881367938641573], 0.3: [0.5335324389233904], 0.4: [0.37913757664223924], 0.5: [0.2250146782355863], 0.6: [0.9388425787227356], 0.7: [0.7864428668970406], 0.8: [0.6361378229518942], 0.9: [0.4883805239623461], 1.0: [0.3422016210182779]}
# # (14_w,28)-aCMA-ES (mu_w=8.1,w_1=21%) in dimension 3658 (seed=2125303054, Fri Jan  2 13:27:42 2026)
# # [{'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.01, 'error': [0.9999821428571428]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.1, 'error': [0.9999821428571428]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.2, 'error': [0.9999107142857143]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.3, 'error': [0.999875]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.4, 'error': [0.9998928571428571]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.5, 'error': [0.9999642857142857]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.6, 'error': [0.9998928571428571]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.7, 'error': [0.9998928571428571]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.8, 'error': [0.9999107142857143]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 0.9, 'error': [0.9999642857142857]}, {'algorithm': 'cmaes', 'dimension': 3658, 'run': 0, 'checkpoint': 1.0, 'error': [0.9999285714285714]}]


# Checkpoint	train_acc_adam  train_acc_cmaes
# 0.01	    0.98377         0.99998214
# 0.10	    0.84309         0.99998214
# 0.20	    0.68814         0.99991071
# 0.30	    0.53353         0.99987500
# 0.40	    0.37914         0.99989286
# 0.50	    0.22501         0.99996429
# 0.60	    0.93884         0.99989286
# 0.70	    0.78644         0.99989286
# 0.80	    0.63614         0.99991071
# 0.90	    0.48838         0.99996429
# 1.00	    0.34220         0.99992857
