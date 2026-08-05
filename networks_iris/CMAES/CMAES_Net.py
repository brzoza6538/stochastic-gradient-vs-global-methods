import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from algorithms import globals
from algorithms import CMAVariation, eswrapper, Eval_wrapper 

import time
import numpy as np

from Net import *
from sklearn.preprocessing import MinMaxScaler



class Evaluation_method():

    def __init__(self, seed, images, labels, popsize ):
        self.popsize = popsize
        self.eval_counter = 0
        self.batch_idx = None 

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=seed) # TODO = add a  seed that changes every time?

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)


        scaler = MinMaxScaler(feature_range=(-1, 1))

        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)


        self.lb = LabelBinarizer()
        self.y_train = self.lb.fit_transform(self.y_train)
        self.y_test = self.lb.transform(self.y_test)




        self.fully_connected_layer1 = FullyConnected(input_size=FULL_IRIS, output_size=HID_LAYER_1)
        self.tanh_layer1 = ReLU()
        self.fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
        self.tanh_layer2 = ReLU()
        self.fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=IRIS_OUTPUT)
        self.tanh_layer3 = Softmax()

        self.my_network = Network(layers=[
                                    self.fully_connected_layer1,
                                    self.tanh_layer1, self.fully_connected_layer2,
                                    self.tanh_layer2, self.fully_connected_layer3,
                                    self.tanh_layer3
                                    ], learning_rate=0.01)


        self.my_loss = Loss(def_loss,def_derivative_loss)

        self.my_network.compile(loss=self.my_loss)
        # self.train_pointer = 0 # HERE


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

        # if self.train_pointer * BATCH_SIZE >= len(self.x_train):
        #     self.train_pointer = 0
             
        # l = int(self.train_pointer) * BATCH_SIZE
        if self.eval_counter % self.popsize == 0:
            self.batch_idx = np.random.choice(
                len(self.x_train),
                BATCH_SIZE,
                replace=False
            )

        self.eval_counter += 1

        batch_x = self.x_train[self.batch_idx]
        batch_y = self.y_train[self.batch_idx]

        # for x_i, y_true in zip(self.x_train[ l :  l + BATCH_SIZE], self.y_train[ l : l + BATCH_SIZE]):
        for x_i, y_true in zip(batch_x, batch_y):
            #x = x.reshape(1, -1)
            y_pred = self.my_network(x_i)

            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            train_loss += np.mean(current_loss)

            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / BATCH_SIZE
        # self.train_pointer += 1 # HERE

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

            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            test_loss += current_loss

            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / len(self.x_test)
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

        test_loss = 0
        correct_predictions = 0

        for x_i, y_true in zip(self.x_test, self.y_test):
            #x = x.reshape(1, -1)
            y_pred = self.my_network(x_i)

            # Calculate loss (you might want to use the proper loss function here)
            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            test_loss += np.mean(current_loss)

            # Check if the prediction is correct
            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / BATCH_SIZE
        # self.train_pointer += 1 # HERE

        print("acccc: ", (accuracy), "lossss: ", (test_loss/BATCH_SIZE))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE
        return (((1 - accuracy), (test_loss/BATCH_SIZE)))# HERE - switch acc and loss

##############






def run_cmaes_net(run_id, images, labels, seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    dimension = ((FULL_IRIS + 1)*HID_LAYER_1 + (HID_LAYER_1 + 1)*HID_LAYER_2 + (HID_LAYER_2 + 1)*IRIS_OUTPUT)
    x0 = np.random.normal(CLAMPS[0], CLAMPS[1], size=dimension)
    # switch_interval = 1
    popsize = int(4 + np.floor(3 * np.log(dimension)))
    eval_meth = Evaluation_method(seed, images, labels, popsize)
    f_eval = Eval_wrapper(eval_meth.evaluate)


    data = eswrapper(
        x=x0,
        fun=f_eval,
        popsize=popsize,
        maxevals=MAX_EVALS,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=None,
    )

    result = []

    max_fes = MAX_EVALS
    for checkpoint in globals.def_checkpoints:
        eval_checkpoint = max_fes * checkpoint

        idx = np.abs(np.array(data.nums_evals) - eval_checkpoint).argmin()


        closest_checkpoint = data.nums_evals[idx]

        if( abs(data.nums_evals[idx] - eval_checkpoint ) < 50 ):
            # closest_value = abs(float(curr_f["global_min"]) - data.midpoint_values[idx])
            print("CHECKPOINT : ", checkpoint)
            closest_value = eval_meth.test_error(data.best_solutions[idx]) # HERE - switch acc and loss

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

    print(run_id)
    return result

