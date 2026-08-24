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
        self.epoch_counter = 0
        
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
        self.tanh_layer1 = Tanh()
        self.fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
        self.tanh_layer2 = Tanh()
        self.fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=IRIS_OUTPUT)
        self.tanh_layer3 = Softmax()

        self.my_network = Network(layers=[
                                    self.fully_connected_layer1,
                                    self.tanh_layer1, self.fully_connected_layer2,
                                    self.tanh_layer2, self.fully_connected_layer3,
                                    self.tanh_layer3
                                    ], learning_rate=0.01)


        self.my_loss = Loss(cross_entropy_loss,cross_entropy_derivative)

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
            self.eval_counter = 0
            if self.batch_idx is None:
                self.batch_idx = np.random.choice(
                    len(self.x_train),
                    min(BATCH_SIZE, len(self.x_train)),
                    replace=False
                )
                self.batch_idx = np.sort(self.batch_idx)

            if self.epoch_counter % 25 == 0:
                self.epoch_counter = 0
                half = BATCH_SIZE // 10

                keep = np.random.choice(self.batch_idx, half, replace=False)

                available = np.setdiff1d(
                    np.arange(len(self.x_train)),
                    self.batch_idx
                )

                new = np.random.choice(available, half, replace=False)

                self.batch_idx = np.sort(np.concatenate([keep, new]))

            self.epoch_counter += 1

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
        
        accuracy = correct_predictions / len(self.batch_idx)

        # print("acccc: ", (accuracy), "lossss: ", (train_loss/len(self.batch_idx))) # HERE learn by acc-loss

        # Calculate average loss and accuracy
        # return (1 - accuracy), len(self.batch_idx)
        return (train_loss/len(self.batch_idx)), len(self.batch_idx)


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
    






    def test_error(self, x, check_time):
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
        
        accuracy = correct_predictions / len(self.x_test)
        # self.train_pointer += 1 # HERE

        print("acccc: ", (accuracy), "lossss: ", (test_loss/len(self.x_test)))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE
        return (((1 - accuracy), (test_loss/len(self.x_test)), check_time))# HERE - switch acc and loss

##############






def run_cmaes_net(run_id, images, labels, seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    dimension = ((FULL_IRIS + 1)*HID_LAYER_1 + (HID_LAYER_1 + 1)*HID_LAYER_2 + (HID_LAYER_2 + 1)*IRIS_OUTPUT)
    x0 = np.random.normal(0.0, 0.1, size=dimension)
    # switch_interval = 1
    popsize = int(4 + np.floor(3 * np.log(dimension)))
    # popsize = int(20 * np.log10(dimension))    

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
        # sigma=0.05
    )

    result = []

    max_fes = MAX_EVALS

    for checkpoint in globals.def_checkpoints:

        print("CHECKPOINT :", checkpoint)



        idx_end = max(1, int(len(data.best_values) * checkpoint))

        help_i = np.argmin(data.best_values[:idx_end])
        sol_data = data.best_solutions[help_i]
        check_time = data.times[help_i]

        test_result = eval_meth.test_error(sol_data, check_time)

        result.append({
            "algorithm": "cmaes",
            "dimension": dimension,
            "run": run_id,
            "checkpoint": checkpoint,
            "error": [test_result]
        })    
    print(run_id)
    return result

