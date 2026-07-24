from algorithms import *
from CMAES import *
from Adam import *
from Adagrad import *
from BFGS import *
from NLShade import *

from functools import partial

RUNS = 1

def gather_data(algorithm, algo_name):
    records = []
    run_records = []


    # for run_id in range(RUNS):
    #     sublist = algorithm(run_id)   # run_adam_net(...)
    #     for item in sublist:
    #         run_records.append(item)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        run_records.extend(
            item
            for sublist in pool.map(algorithm, range(RUNS))
            for item in sublist
        )

    record = {checkpoint: [] for checkpoint in def_checkpoints}
    for entry in run_records:
        record[entry["checkpoint"]].append(entry["error"])

    for checkpoint in def_checkpoints:
        errors_at_checkpoint = record[checkpoint]

        mean = np.mean(errors_at_checkpoint)
        std = np.std(errors_at_checkpoint)
        median = np.median(errors_at_checkpoint)
        minimum = np.min(errors_at_checkpoint)
        maximum = np.max(errors_at_checkpoint)

        mean = mean if mean >= def_smallest_val else 0
        std = std if std >= def_smallest_val else 0
        median = median if median >= def_smallest_val else 0
        minimum = minimum if minimum >= def_smallest_val else 0
        maximum = maximum if maximum >= def_smallest_val else 0

        records.append({
            "checkpoint": checkpoint,
            "mean": mean,
            "std": std,
            "median": median,
            "max": maximum,
            "min": minimum,
        })

    try:
        os.mkdir(f'{algo_name}_logs')
        print(f"Directory {algo_name}_logs created successfully.")
    except PermissionError:
        print(f"Permission denied: Unable to create {algo_name}_logs.")
    except Exception as e:
        print(f"An error occurred: {e}")

    keys = records[0].keys()

    with open(f'./{algo_name}_logs/{algo_name}_records.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    run_keys = run_records[0].keys()
    with open(f'./{algo_name}_logs/{algo_name}_run_records.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=run_keys)
        writer.writeheader()
        writer.writerows(run_records)

