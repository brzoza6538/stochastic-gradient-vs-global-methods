import numpy as np
from algorithms import globals
from algorithms import CMAES
import time

def run_CMAES(dimension, curr_f, run_id, seed=None, dif_lambd=False):
    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)
    np.random.seed(seed)

    result = []

    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    if dif_lambd:
        lamb = lamb or int(4 * dimension)
    else:
        lamb = lamb or int(4 + np.floor(3 * np.log(dimension)))

    if(run_id == 0 or run_id == 3):
        print(x)

    alg = CMAES(eval.evaluate, dimension, mean=x, lamb=lamb)
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


globals.gather_data(run_CMAES, "cmaes_clamp")
globals.gather_data(run_CMAES, "cmaes_clamp_lambd=4m", dif_lambd=True)