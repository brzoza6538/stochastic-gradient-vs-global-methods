#!/bin/bash

sbatch ./Adam/run.sh
sbatch ./Adagrad/run.sh
sbatch ./BFGS/run.sh
sbatch ./cmaes/run.sh
sbatch ./nl_shade/run.sh

# squeue -u $USER
# scancel 
# scontrol show job 20860381
