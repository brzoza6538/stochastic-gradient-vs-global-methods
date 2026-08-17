#!/bin/bash -l
#SBATCH -J BFGS_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_iris/BFGS/BFGS_output.log"
#SBATCH --error="networks_iris/BFGS/BFGS_error.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_iris"
cd networks_iris/BFGS

export PYTHONUNBUFFERED=1
python3 -u BFGS_script.py