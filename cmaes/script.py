import numpy as np
from algorithms import globals
from algorithms import CMAVariation, eswrapper, Eval_wrapper 

import time
from functools import partial
import numpy as np


def run_cmaes(dimension, curr_f, run_id,  seed=None):

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    x0 = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)

    # switch_interval = 1
    popsize = int(4 + np.floor(3 * np.log(dimension)))

    f_eval = Eval_wrapper(globals.Evaluation_method(curr_f, dimension).evaluate)


    data = eswrapper(
        x=x0,
        fun=f_eval,
        popsize=popsize,
        maxevals=globals.def_max_fes * dimension,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=None,
    )

    result = []

    max_fes = globals.def_max_fes * dimension
    for checkpoint in globals.def_checkpoints:
        eval_checkpoint = max_fes * checkpoint

        idx = np.abs(data.nums_evals - eval_checkpoint).argmin()


        closest_checkpoint = data.nums_evals[idx]

        if( abs(data.nums_evals[idx] - eval_checkpoint ) < 50 ):
            # closest_value = abs(float(curr_f["global_min"]) - data.midpoint_values[idx])
            closest_value = data.best_values[idx]
            timer = data.times[idx]

            result.append({
                "function": curr_f["shortname"],
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value, timer]
            })
        else:
            closest_value = 0
            timer = data.times[idx]

            result.append({
                "function": curr_f["shortname"],
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value, timer]
            })
    return result

globals.gather_data(partial(run_cmaes), "cmaes_curr_updt")


# dimension = 50
# results = run_cmaes(dimension, globals.CEC2013[0], 0)
# print(results)
