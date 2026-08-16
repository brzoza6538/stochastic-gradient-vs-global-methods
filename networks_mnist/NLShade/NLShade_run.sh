#!/bin/bash -l
#SBATCH -J M_NLShade_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_mnist/NLShade/NLShade_output.log"
#SBATCH --error="networks_mnist/NLShade/NLShade_error.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_mnist"
cd networks_mnist/NLShade

export PYTHONUNBUFFERED=1
python3 -u NLShade_script.py