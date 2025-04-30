import numpy as np

from algorithms import globals
from algorithms import Adam

def run_Adam(dimension, curr_f, run_id):
    result = []
    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    alg = Adam(eval.evaluate, eval.gradient, dimension, x=x)
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

globals.gather_data(run_Adam, "adam")