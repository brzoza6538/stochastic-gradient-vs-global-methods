#!/bin/bash -l
#SBATCH -J M_CMAES_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_mnist/CMAES/CMAES_output_2.log"
#SBATCH --error="networks_mnist/CMAES/CMAES_error_2.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_mnist"
cd networks_mnist/CMAES
export PYTHONUNBUFFERED=1
python3 -u CMAES_script.py