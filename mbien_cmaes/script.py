import numpy as np
from algorithms import globals
from algorithms import CMAVariation, CMAExperimentCallback, lincmaes, OptFun, eswrapper

import time
from functools import partial
import numpy as np
from opfunu.cec_based.cec2014 import F12014  # przykład funkcji CEC (jeśli chcesz)
from typing import Optional

# def run_CMAES(dimension, curr_f, run_id, seed=None, dif_lambd=False):

def run_cmaes(dimension, curr_f, run_id,  seed=None):
    f_eval = curr_f["func"](ndim=dimension)

    seed = seed or int((time.time() * 1000) + run_id)  # Generujemy nasiono na podstawie czasu i run_id
    seed = seed % (2**32)

    x0 = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)

    # switch_interval = 1   # mój kod nie ma switch interval
    popsize = int(4 + np.floor(3 * np.log(dimension)))

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
            closest_value = abs(float(curr_f["global_min"]) - data.best_values[idx])
            result.append({
                "function": curr_f["shortname"],
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value]
            })
        else:
            closest_value = 0
            result.append({
                "function": curr_f["shortname"],
                "dimension": dimension,
                "run": run_id,
                "checkpoint": checkpoint,
                "error": [closest_value]
            })


    return result

globals.gather_data(partial(run_cmaes), "mbien_cmaes")

