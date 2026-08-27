#!/bin/bash -l
#SBATCH -J adagrad_f_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_mnist/Adagrad_fair/adagrad_output_1.log"
#SBATCH --error="networks_mnist/Adagrad_fair/adagrad_error_1.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_mnist"
cd networks_mnist/Adagrad_fair
export PYTHONUNBUFFERED=1
python3 -u Adagrad_fair_script.py