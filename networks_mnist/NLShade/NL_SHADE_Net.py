
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from algorithms import globals

import time
import numpy as np

from Net import *
from sklearn.preprocessing import MinMaxScaler
from algorithms import NL_SHADE_RSP_MID


class Evaluation_method():

    def __init__(self, seed, images, labels ):
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

        self.E_layer = EmbedLayer(input_size=FULL_MNIST, output_size=INPUT)
        self.tanh_layer0 = Tanh()

        self.fully_connected_layer1 = FullyConnected(input_size=INPUT, output_size=HID_LAYER_1)
        self.tanh_layer1 = Tanh()
        self.fully_connected_layer2 = FullyConnected(input_size=HID_LAYER_1, output_size=HID_LAYER_2)
        self.tanh_layer2 = Tanh()
        self.fully_connected_layer3 = FullyConnected(input_size=HID_LAYER_2, output_size=MNIST_OUTPUT)
        self.tanh_layer3 = Softmax()

        self.my_network = Network(layers=[self.E_layer, self.tanh_layer0,
                                    self.fully_connected_layer1,
                                    self.tanh_layer1, self.fully_connected_layer2,
                                    self.tanh_layer2, self.fully_connected_layer3,
                                    self.tanh_layer3
                                    ], learning_rate=0.01)


        self.my_loss = Loss(def_loss,def_derivative_loss)

        self.my_network.compile(loss=self.my_loss)
        self.train_pointer = 0

        self.wrapper = None
        self.eval_counter = 0
        self.batch_idx = None
        self.epoch_counter = 0


    def evaluate(self,x):
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



        if self.eval_counter == 0:
            if self.batch_idx is None:
                self.batch_idx = np.random.choice(
                    len(self.x_train),
                    min(BATCH_SIZE, len(self.x_train)),
                    replace=False
                )
                self.batch_idx = np.sort(self.batch_idx)

            elif self.epoch_counter % 100 == 0:
                self.epoch_counter = 0
                half = BATCH_SIZE // 5

                keep = np.random.choice(self.batch_idx, half, replace=False)

                available = np.setdiff1d(
                    np.arange(len(self.x_train)),
                    self.batch_idx
                )

                new = np.random.choice(available, half, replace=False)

                self.batch_idx = np.sort(np.concatenate([keep, new]))

            self.epoch_counter += 1


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
        # self.train_pointer += 0.0001 # HERE

        # print("acccc: ", (accuracy), "lossss: ", (train_loss/len(self.batch_idx)))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE


        self.eval_counter += 1

        if self.eval_counter >= self.wrapper.pop_size:
            self.eval_counter = 0



        return train_loss / len(self.batch_idx), len(self.batch_idx)

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

            current_loss = self.my_loss.calculate_loss(y_true, y_pred)
            test_loss += np.mean(current_loss)

            predicted_label = np.argmax(y_pred)
            true_label = np.argmax(y_true)
            correct_predictions += (predicted_label == true_label)
        
        accuracy = correct_predictions / len(self.x_test)
        # self.test_pointer += 0.0001 # HERE

        print("acccc: ", (accuracy), "lossss: ", (test_loss/len(self.x_test)))

        # Calculate average loss and accuracy
        # return (1 - accuracy), BATCH_SIZE
        return (((1 - accuracy), (test_loss/len(self.x_test)), check_time)) # HERE - switch acc and loss







def run_nlshade_net(run_id, images, labels, seed=None):

    seed = seed or int((time.time() * 1000) + run_id)
    seed = seed % (2**32)

    dimension = ((INPUT + 1)*HID_LAYER_1 +
                 (HID_LAYER_1 + 1)*HID_LAYER_2 +
                 (HID_LAYER_2 + 1)*MNIST_OUTPUT)

    # pop_size = dimension * 5
    # pop_size = int(4 + np.floor(3 * np.log(dimension)))
    pop_size = int(20 * np.log10(dimension))    
    x0 = np.random.normal(0.0, 0.5, size=(pop_size, dimension))

    eval_meth = Evaluation_method(seed, images, labels)
    f_eval = eval_meth.evaluate

    algo = NL_SHADE_RSP_MID(
        f_objective=f_eval,
        dimension=dimension,
        X=x0,
        objective_limit=MAX_EVALS,
        min_clamp=globals.def_clamps[0],
        max_clamp=globals.def_clamps[1],
        checkpoints=globals.def_checkpoints,
        pop_size=pop_size,
    )
    eval_meth.wrapper = algo

    print("START  :  " , time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    algo.start()

    result = []
    max_fes = MAX_EVALS

    for checkpoint in globals.def_checkpoints:
        eval_checkpoint = max_fes * checkpoint

        if len(algo.log[checkpoint]) > 0:
            closest_value = algo.log[checkpoint][0]
            check_time = algo.log[checkpoint][1]
            checkpoint_x = algo.help_log[checkpoint][0]
            loss_grad = eval_meth.test_error(checkpoint_x, check_time)

            result.append({
                "algorithm": "nlshade",
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [loss_grad]
            })

        else:
            result.append({
                "algorithm": "nlshade",
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [(None, None)]#result[-1]["error"]]
            })

    print(run_id)
    return result
