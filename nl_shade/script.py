import numpy as np
from algorithms import globals
from algorithms import NL_SHADE_RSP_MID

import time
from functools import partial

def run_nl_shade(dimension, curr_f, run_id, seed=None):
    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    np.random.seed(seed)
    
    result = []

    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    alg = NL_SHADE_RSP_MID(f_objective=eval.evaluate, dimension=dimension)

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


globals.gather_data(partial(run_nl_shade), "nl_shade_curr_updt")
