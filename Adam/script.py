import numpy as np
from algorithms import globals
from algorithms import Adam
from algorithms import NormAdam

import time
from functools import partial

def run_Adam(dimension, curr_f, run_id, seed=None, lr=0.01, B1=0.9, B2=0.999):
    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    np.random.seed(seed)
    
    result = []

    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    alg = Adam(eval.evaluate, eval.gradient, dimension, x=x, lr=lr, B1=B1, B2=B2)
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

globals.gather_data(partial(run_Adam, lr=0.01, B1=0.9, B2=0.999), "adam_clamp_lr=0.01_B1=0.9_B2=0.999_t2")

globals.gather_data(partial(run_Adam, lr=0.01, B1=0.9, B2=0.999), "adam_clamp_lr=0.01_B1=0.9_B2=0.999")
globals.gather_data(partial(run_Adam, lr=0.01, B1=0.8, B2=0.99), "adam_clamp_lr=0.01_B1=0.8_B2=0.99")
globals.gather_data(partial(run_Adam, lr=0.001, B1=0.9, B2=0.999), "adam_clamp_lr=0.001_B1=0.9_B2=0.999")
globals.gather_data(partial(run_Adam, lr=0.001, B1=0.8, B2=0.99), "adam_clamp_lr=0.001_B1=0.8_B2=0.99")


# globals.gather_data(partial(run_Adam, lr=0.01, B1=0.9, B2=0.999), "adam_norm_graddiv_lr=0.01_B1=0.9_B2=0.999")
