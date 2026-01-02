#!/bin/bash -l
#SBATCH -J Adagrad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid	
#SBATCH --output="Adagrad/output.log"
#SBATCH --error="Adagrad/error.log"


BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$BASE_DIR
cd Adagrad
python3 script.py