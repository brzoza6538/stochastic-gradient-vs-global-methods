#!/bin/bash -l
#SBATCH -J CMAES
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid	
#SBATCH --output="CMAES/output.log"
#SBATCH --error="CMAES/error.log"


BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$BASE_DIR
cd CMAES
python3 script.py