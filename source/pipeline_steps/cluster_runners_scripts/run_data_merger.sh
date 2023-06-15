#!/bin/bash
#SBATCH -J MERGE
#SBATCH -o /ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/logs/data_merger_%j.log
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --threads-per-core=1
#SBATCH --mem=2G
# SBATCH --mail-type=END

# Parsing input args into variables
PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/anaconda3/bin/activate local_env
echo "local_env env activated"

CODE_DIR="/ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/temp_new_pipeline/FromExperimentToAnalysis/source/pipeline_steps"
cd $CODE_DIR
python data_merger.py "$PARAMS_FILE"