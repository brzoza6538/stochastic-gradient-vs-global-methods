#!/bin/bash -l
#SBATCH -J CMAES
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid	
#SBATCH --output="Adam/output.log"
#SBATCH --error="Adam/error.log"

cd $SLURM_SUBMIT_DIR
source ~/take_1/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/take_1
cd CMAES
python3 script.py