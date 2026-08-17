#!/bin/bash -l
#SBATCH -J CMAES_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_iris/CMAES/CMAES_output.log"
#SBATCH --error="networks_iris/CMAES/CMAES_error.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_iris"
cd networks_iris/CMAES
python3 CMAES_script.py