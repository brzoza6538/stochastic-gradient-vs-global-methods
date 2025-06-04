SLURM_SUBMIT_DIR="/home/plgrid/plgmichalbrz/take_1"
BASE_DIR=~/$(basename $SLURM_SUBMIT_DIR)

cd $SLURM_SUBMIT_DIR
source $BASE_DIR/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$BASE_DIR
cd mbien_cmaes
python3 script.py