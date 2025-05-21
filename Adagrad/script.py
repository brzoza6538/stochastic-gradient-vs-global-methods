import numpy as np
from algorithms import globals
from algorithms import Adagrad
import time
from functools import partial


def run_Adagrad(dimension, curr_f, run_id, seed=None, lr=0.001):
    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    np.random.seed(seed)
    
    result = []

    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    alg = Adagrad(eval.evaluate, eval.gradient, dimension, x=x, B=lr)

    alg.start()
    log = alg.log
    for checkpoint in log.keys():
        result.append({
            "function": curr_f["shortname"],
            "dimension": dimension,
            "run": run_id,
            "checkpoint": checkpoint,
            "error": log[checkpoint]
            })
    return result


globals.gather_data(partial(run_Adagrad, lr=0.001), "adagrad_clamp_lr=0.001")
globals.gather_data(partial(run_Adagrad, lr=0.01), "adagrad_clamp_lr=0.01")
globals.gather_data(partial(run_Adagrad, lr=0.1), "adagrad_clamp_lr=0.1")
