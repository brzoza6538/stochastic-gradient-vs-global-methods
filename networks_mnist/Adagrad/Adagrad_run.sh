#!/bin/bash -l
#SBATCH -J Adagrad_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_mnist/Adagrad/Adagrad_output_acloss_batched.log"
#SBATCH --error="networks_mnist/Adagrad/Adagrad_error_acloss_batched.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_mnist"
cd networks_mnist/Adagrad
python3 Adagrad_script.py