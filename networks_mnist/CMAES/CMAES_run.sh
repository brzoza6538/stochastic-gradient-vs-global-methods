#!/bin/bash -l
#SBATCH -J CMAES_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=15GB
#SBATCH --time=48:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_mnist/CMAES/CMAES_output.log"
#SBATCH --error="networks_mnist/CMAES/CMAES_error.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_mnist"
cd networks_mnist/CMAES
python3 CMAES_script.py