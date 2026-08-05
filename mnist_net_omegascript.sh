#!/bin/bash

sbatch ./networks_mnist/Adagrad/Adagrad_run.sh
sbatch ./networks_mnist/Adam/Adam_run.sh
sbatch ./networks_mnist/CMAES/CMAES_run.sh
sbatch ./networks_mnist/BFGS/BFGS_run.sh
sbatch ./networks_mnist/NLShade/NLShade_run.sh

# squeue -u $USER
# scancel 
# scontrol show job 20860381
