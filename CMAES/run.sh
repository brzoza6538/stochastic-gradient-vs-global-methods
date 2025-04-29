#!/bin/bash -l
#SBATCH -J CMAES
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH -p plgrid	
#SBATCH --output="CMAES/output.log"
#SBATCH --error="CMAES/error.log"

cd $SLURM_SUBMIT_DIR
source ~/take_1/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/take_1
cd CMAES
python3 script.py