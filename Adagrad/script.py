import numpy as np
from algorithms import globals
from algorithms import Adagrad
import time

def run_Adagrad(dimension, curr_f, run_id, seed=None, lr=None):
    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    np.random.seed(seed)
    
    result = []

    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    if (lr is not None):
        alg = Adagrad(eval.evaluate, eval.gradient, dimension, x=x, lr=lr)
    else:
        alg = Adagrad(eval.evaluate, eval.gradient, dimension, x=x)
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

globals.gather_data(run_Adagrad, "adagrad_clamp_lr=0.001")
globals.gather_data(run_Adagrad, "adagrad_clamp_lr=0.01")