import numpy as np
import csv
import multiprocessing as mp
from algorithms import globals
from algorithms import Adam

def run_adam(dimension, curr_f, run_id):
    result = []
    x = np.random.uniform(globals.def_clamps[0], globals.def_clamps[1], size=dimension)
    eval = globals.Evaluation_method(curr_f, dimension)

    alg = Adam(eval.evaluate, eval.gradient, dimension, x=x)
    alg.start()
    log = alg.log
    #log examlpe {0.01: [np.float64(160513.26014398824)], 0.1: [np.float64(83145.16792425113)], 0.2: [np.float64(33463.99114162416)], 0.3: [np.float64(9418.598537377286)], 0.4: [np.float64(581.3323467438954)], 0.5: [0], 0.6: [0], 0.7: [0], 0.8: [0], 0.9: [0], 1.0: [0]}
    for checkpoint in log.keys():
        result.append({
            "function": curr_f["shortname"],
            "dimension": dimension,
            "run": run_id,
            "checkpoint": checkpoint,
            "error": log[checkpoint]
            })
    return result


for curr_f in globals.CEC2013:
    Adam_records = []
    Adam_run_records = []
    for dimension in globals.def_dimensions:

        with mp.Pool(processes=mp.cpu_count()) as pool:
            Adam_run_records.extend([item for sublist in pool.starmap(
                run_adam,
                [(dimension, curr_f, run_id) for run_id in range(globals.def_runs)]
            ) for item in sublist])

        record = {checkpoint: [] for checkpoint in globals.def_checkpoints}
        for entry in Adam_run_records:
            if(entry["dimension"] == dimension):
                record[entry["checkpoint"]].append(entry["error"])

        for checkpoint in globals.def_checkpoints:
            errors_at_checkpoint = record[checkpoint]
            mean = np.mean(errors_at_checkpoint)
            std = np.std(errors_at_checkpoint)
            median = np.median(errors_at_checkpoint)
            minimum = np.min(errors_at_checkpoint)
            maximum = np.max(errors_at_checkpoint)

            Adam_records.append({
                "function": curr_f["shortname"],
                "dimensions": dimension,
                "checkpoint": checkpoint,
                "mean": mean,
                "std": std,
                "median": median,
                "max": maximum,
                "min": minimum,
            })
    keys = Adam_records[0].keys()
    with open(f'Adam_records_{curr_f["shortname"]}.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(Adam_records)

    run_keys = Adam_run_records[0].keys()
    with open(f'Adam_run_records_{curr_f["shortname"]}.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=run_keys)
        writer.writeheader()
        writer.writerows(Adam_run_records)