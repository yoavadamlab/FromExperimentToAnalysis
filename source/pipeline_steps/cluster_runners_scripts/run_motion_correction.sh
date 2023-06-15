#!/bin/bash
#SBATCH -J MC
#SBATCH -o /ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/logs/motion_correction_%j.log
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --threads-per-core=1
#SBATCH --mem=256G
#SBATCH --mail-type=END

# Parsing input args into variables
PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/anaconda3/bin/activate caiman
echo "caiman env activated"

CODE_DIR="/ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/temp_new_pipeline/FromExperimentToAnalysis/source/pipeline_steps"
cd $CODE_DIR
python motion_correction.py "$PARAMS_FILE"
