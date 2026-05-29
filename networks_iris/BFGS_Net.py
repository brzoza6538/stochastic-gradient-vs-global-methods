from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from algorithms import globals
from algorithms import CMAVariation, eswrapper, Eval_wrapper 

import time

from Net import *
from sklearn.preprocessing import MinMaxScaler
from algorithms.BFGS import BFGS





class Evaluation_method():

    def __init__(self, seed, images, labels ):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=seed) # TODO = add a  seed that changes every time?

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # Normalize pixel values to be between -1 and 1

        scaler = MinMaxScaler(feature_range=(-1, 1))

        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        # Flatten the images
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)


        # Convert class vectors to binary class matrices
        self.lb = LabelBinarizer()
        self.y_train = self.lb.fit_transform(self.y_train)
        self.y_test = self.lb.transform(self.y_test)




        # Instantiate layers
        self.fully_connected_layer1 = FullyConnected(input_size=FULL_IRIS, output_size=HID_LAYER_1)
        self.tanh_layer1 = Tanh()
        self.fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
        self.tanh_layer2 = Tanh()
        self.fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=IRIS_OUTPUT)
        self.tanh_layer3 = Tanh()

        self.my_network = Network(layers=[
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

        if int(self.train_pointer) * BATCH_SIZE > len(self.x_train):
            self.train_pointer = 0

        l = int(self.train_pointer) * BATCH_SIZE



        for x_i, y_true in zip(self.x_train[ l :  l + BATCH_SIZE], self.y_train[ l : l + BATCH_SIZE]):
            #x = x.reshape(1, -1)
            y_pred = np.maximum(self.my_network(x_i), 0)

            # Calculate loss (you might want to use the proper loss function here)
            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            train_loss += np.mean(current_loss)

            # Check if the prediction is correct
            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / BATCH_SIZE
        # self.train_pointer += 0.0001 # HERE

        print("acccc: ", (accuracy), "lossss: ", (train_loss/BATCH_SIZE))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE
        return (train_loss/BATCH_SIZE), BATCH_SIZE


    def test(self, x):
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

        test_loss = 0
        correct_predictions = 0

        for x_i, y_true in zip(self.x_test, self.y_test):
            #x = x.reshape(1, -1)
            y_pred = self.my_network(x_i)

            # Calculate loss (you might want to use the proper loss function here)
            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            test_loss += current_loss

            # Check if the prediction is correct
            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / len(self.x_test)
        # Calculate average loss and accuracy
        return accuracy
    
    

    def test_error(self, x):
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

        if int(self.train_pointer) * BATCH_SIZE > len(self.x_train):
            self.train_pointer = 0

        l = int(self.train_pointer) * BATCH_SIZE



        for x_i, y_true in zip(self.x_train[ l :  l + BATCH_SIZE], self.y_train[ l : l + BATCH_SIZE]):
            #x = x.reshape(1, -1)
            y_pred = np.maximum(self.my_network(x_i), 0)

            # Calculate loss (you might want to use the proper loss function here)
            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            train_loss += np.mean(current_loss)

            # Check if the prediction is correct
            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / BATCH_SIZE
        # self.train_pointer += 0.0001 # HERE

        print("acccc: ", (accuracy), "lossss: ", (train_loss/BATCH_SIZE))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE
        return (train_loss/BATCH_SIZE)


# ---------- Finite difference gradient ----------
def finite_diff_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    fx = f(x)

    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (f(x_eps) - fx) / eps

    return grad






# ---------- Wrapper objective ----------
class BFGSObjectiveWrapper:
    def __init__(self, eval_meth):
        self.eval_meth = eval_meth

    def f_objective(self, x):
        loss, evals = self.eval_meth.evaluate(x)
        return loss, evals

    def f_gradient(self, x):
        grad = finite_diff_grad(lambda v: self.eval_meth.evaluate(v)[0], x)
        return grad, len(x) * 2  # rough eval count


# ---------- Main runner ----------
def run_bfgs_net(run_id, images, labels, seed=None):

    seed = seed or int((time.time() * 1000) + run_id)
    seed = seed % (2**32)

    # ---------- Data ----------
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=seed
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # ---------- Evaluation ----------
    eval_meth = Evaluation_method(seed, images, labels)
    wrapper = BFGSObjectiveWrapper(eval_meth)

    # ---------- Dimension ----------
    dimension = (
        (FULL_IRIS + 1) * HID_LAYER_1 +
        (HID_LAYER_1 + 1) * HID_LAYER_2 +
        (HID_LAYER_2 + 1) * IRIS_OUTPUT
    )

    x0 = np.random.uniform(CLAMPS[0], CLAMPS[1], size=dimension)

    # ---------- Optimizer ----------
    optimizer = BFGS(
        f_objective=wrapper.f_objective,
        f_gradient=wrapper.f_gradient,
        dimension=dimension,
        x=x0,
        max_fes=MAX_EVALS,
        min_clamp=globals.def_clamps[0],
        max_clamp=globals.def_clamps[1],
        checkpoints=globals.def_checkpoints
    )

    optimizer.start()

    best_x = optimizer.x

    # ---------- Evaluation on checkpoints ----------
    result = []
    max_fes = MAX_EVALS

    for checkpoint in globals.def_checkpoints:
        if checkpoint in optimizer.log and len(optimizer.log[checkpoint]) > 0:
            checkpoint_data = optimizer.log[checkpoint][-1]
            checkpoint_x = checkpoint_data["x"]
            loss_grad = eval_meth.test_error(checkpoint_x)

        else:
            loss_grad = 0

        result.append({
            "algorithm": "bfgs",
            "dimension": dimension,
            "run": run_id,
            "checkpoint": checkpoint,
            "error": [loss_grad]
        })

    print("Run:", run_id)
    return result