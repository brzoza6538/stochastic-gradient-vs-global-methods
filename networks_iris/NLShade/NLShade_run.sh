#!/bin/bash -l
#SBATCH -J NLShade_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid
#SBATCH --output="networks_iris/NLShade/NLShade_output.log"
#SBATCH --error="networks_iris/NLShade/NLShade_error.log"



BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}:${BASE_DIR}/networks_iris"
cd networks_iris/NLShade

export PYTHONUNBUFFERED=1
python3 -u NLShade_script.py