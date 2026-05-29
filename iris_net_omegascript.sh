#!/bin/bash

sbatch ./networks_iris/Adagrad_run.sh
sbatch ./networks_iris/Adam_run.sh
sbatch ./networks_iris/CMAES_run.sh
sbatch ./networks_iris/BFGS_run.sh
sbatch ./networks_iris/NLShade_run.sh

# squeue -u $USER
# scancel 
# scontrol show job 20860381
