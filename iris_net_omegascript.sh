#!/bin/bash

sbatch ./networks_iris/Adagrad/Adagrad_run.sh
sbatch ./networks_iris/Adam/Adam_run.sh
sbatch ./networks_iris/CMAES/CMAES_run.sh
sbatch ./networks_iris/BFGS/BFGS_run.sh
sbatch ./networks_iris/NLShade/NLShade_run.sh

# squeue -u $USER
# scancel 
# scontrol show job 20860381
