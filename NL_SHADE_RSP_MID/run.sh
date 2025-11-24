#!/bin/bash -l
#SBATCH -J NL_SHADE_RSP_MID
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid	
#SBATCH --output="NL_SHADE_RSP_MID/output.log"
#SBATCH --error="NL_SHADE_RSP_MID/error.log"


BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$BASE_DIR
cd NL_SHADE_RSP_MID
python3 brudnopis.py