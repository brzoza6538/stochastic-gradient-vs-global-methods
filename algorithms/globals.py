import opfunu.cec_based as cec
import numpy as np 
import csv
import multiprocessing as mp

def_dimensions = [10, 30, 50]
def_runs = 51
def_max_fes = 10000
def_checkpoints = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
def_smallest_val = 1e-8
def_clamps = [-100, 100]
 
class Evaluation_method():
    def __init__(self, tested_f, dimension):
        self.tested_f = tested_f
        self.objective_f = self.tested_f["func"](ndim=dimension)

    def evaluate(self, x):
        Y = self.objective_f.evaluate(x)
        error = abs(Y - self.tested_f["global_min"])
        evaluations_used = 1
        return error, evaluations_used

    def gradient(self, x, E=1e-8):
        grad = np.zeros_like(x)
        fx = self.objective_f.evaluate(x)
        
        for i in range(len(x)):
            x_eps = np.array(x, copy=True)
            x_eps[i] += E
            grad[i] = (self.objective_f.evaluate(x_eps) - fx) / E
        
        evaluations_used = len(x)
        return grad, evaluations_used


def gather_data(algorithm, algo_name):
    for curr_f in CEC2013:
        print("doing f ", curr_f["shortname"])
        records = []
        run_records = []
        for dimension in def_dimensions:
            print("\tdimension ", dimension)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                run_records.extend([item for sublist in pool.starmap(
                    algorithm,
                    [(dimension, curr_f, run_id) for run_id in range(def_runs)]
                ) for item in sublist])

            record = {checkpoint: [] for checkpoint in def_checkpoints}
            for entry in run_records:
                if(entry["dimension"] == dimension):
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
                    "function": curr_f["shortname"],
                    "dimensions": dimension,
                    "checkpoint": checkpoint,
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "max": maximum,
                    "min": minimum,
                })

        keys = records[0].keys()
        with open(f'./{algo_name}_logs/{algo_name}_records_{curr_f["shortname"]}.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

        run_keys = run_records[0].keys()
        with open(f'./{algo_name}_logs/{algo_name}_run_records_{curr_f["shortname"]}.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_keys)
            writer.writeheader()
            writer.writerows(run_records)

CEC2013 = [
    {"shortname": "F12013", "name": "Sphere Function", "func": cec.F12013, "global_min": -1400},
    {"shortname": "F22013", "name": "Rotated High Conditioned Elliptic Function", "func": cec.F22013, "global_min": -1300},
    {"shortname": "F32013", "name": "Rotated Bent Cigar Function", "func": cec.F32013, "global_min": -1200},
    {"shortname": "F42013", "name": "Rotated Discus Function", "func": cec.F42013, "global_min": -1100},
    {"shortname": "F52013", "name": "Different Powers Function", "func": cec.F52013, "global_min": -1000},
    {"shortname": "F62013", "name": "Rotated Rosenbrock’s Function", "func": cec.F62013, "global_min": -900},
    {"shortname": "F72013", "name": "Rotated Schaffers F7 Function", "func": cec.F72013, "global_min": -800},
    {"shortname": "F82013", "name": "Rotated Ackley’s Function", "func": cec.F82013, "global_min": -700},
    {"shortname": "F92013", "name": "Rotated Weierstrass Function", "func": cec.F92013, "global_min": -600},
    {"shortname": "F102013", "name": "Rotated Griewank’s Function", "func": cec.F102013, "global_min": -500},
    {"shortname": "F112013", "name": "Rastrigin’s Function", "func": cec.F112013, "global_min": -400},
    {"shortname": "F122013", "name": "Rotated Rastrigin’s Function", "func": cec.F122013, "global_min": -300},
    {"shortname": "F132013", "name": "Non-Continuous Rotated Rastrigin’s Function", "func": cec.F132013, "global_min": -200},
    {"shortname": "F142013", "name": "Schwefel's Function", "func": cec.F142013, "global_min": -100},
    {"shortname": "F152013", "name": "Rotated Schwefel's Function", "func": cec.F152013, "global_min": 100},
    {"shortname": "F162013", "name": "Rotated Katsuura Function", "func": cec.F162013, "global_min": 200},
    {"shortname": "F172013", "name": "Lunacek Bi_Rastrigin Function", "func": cec.F172013, "global_min": 300},
    {"shortname": "F182013", "name": "Rotated Lunacek Bi_Rastrigin Function", "func": cec.F182013, "global_min": 400},
    {"shortname": "F192013", "name": "Expanded Griewank’s plus Rosenbrock’s Function", "func": cec.F192013, "global_min": 500},
    {"shortname": "F202013", "name": "Expanded Scaffer’s F6 Function", "func": cec.F202013, "global_min": 600},
    {"shortname": "F212013", "name": "Composition Function 1 (n=5,Rotated)", "func": cec.F212013, "global_min": 700},
    {"shortname": "F222013", "name": "Composition Function 2 (n=3,Unrotated)", "func": cec.F222013, "global_min": 800},
    {"shortname": "F232013", "name": "Composition Function 3 (n=3,Rotated)", "func": cec.F232013, "global_min": 900},
    {"shortname": "F242013", "name": "Composition Function 4 (n=3,Rotated)", "func": cec.F242013, "global_min": 1000},
    {"shortname": "F252013", "name": "Composition Function 5 (n=3,Rotated)", "func": cec.F252013, "global_min": 1100},
    {"shortname": "F262013", "name": "Composition Function 6 (n=5,Rotated)", "func": cec.F262013, "global_min": 1200},
    {"shortname": "F272013", "name": "Composition Function 7 (n=5,Rotated)", "func": cec.F272013, "global_min": 1300},
    {"shortname": "F282013", "name": "Composition Function 8 (n=5,Rotated)", "func": cec.F282013, "global_min": 1400},
]