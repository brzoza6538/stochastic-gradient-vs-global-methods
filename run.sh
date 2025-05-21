#!/bin/bash

sbatch ./CMAES/run.sh
sbatch ./Adam/run.sh
sbatch ./Adagrad/run.sh

# squeue -u $USER
# scancel 
# scontrol show job 20860381
