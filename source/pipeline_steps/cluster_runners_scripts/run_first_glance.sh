#!/bin/bash
#SBATCH -J FG
#SBATCH -o /ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/logs/first_glance_%j.log
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --threads-per-core=1
#SBATCH --mem=256G
#SBATCH --mail-type=END

GUI_PARAMS_PATH=${1:-"None"}


CODE_DIR="/ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/temp_new_pipeline/FromExperimentToAnalysis/source/pipeline_steps"
cd $CODE_DIR
/usr/local/bin/matlab -nodesktop -nosplash -noFigureWindows -batch "first_glance('"$GUI_PARAMS_PATH"'); exit"

