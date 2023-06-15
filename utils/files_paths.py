import sys
import os
pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### BASH runners ### 
CLUSTER_RUNNERS_DIR = '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Code/temp_new_pipeline/FromExperimentToAnalysis/source/pipeline_steps/cluster_runners_scripts/'
RAW_TRACES_EXTRACTION_BASH = 'run_first_glance.sh'
MOTION_CORRECTION_BASH = 'run_motion_correction.sh'
SPATIAL_FOOTPRINT_BASH = 'run_spatial_footprint.sh'
SPIKE_DETECTION_BASH = 'run_spike_detection.sh'
BEHAVEIOR_TRACES_MERGER_BASH = 'run_data_merger.sh'
### 
VIRMEN_DIR = "/ems/elsc-labs/adam-y/Adam-Lab-Shared/Mice_Training/imaging_data/"
DATASET_DIR = "/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/Behavior_and_Imagging/"
VIRMEN_DIR_WINDOWS = r"Z:\Adam-Lab-Shared\Mice_Training\imaging_data"
DATASET_DIR_WINDOWS = r"Z:\Adam-Lab-Shared\Data\Behavior_and_Imagging"
DB_UPLOAD_QUEUE_PATH = os.path.join(pipeline_dir, "metadata", "DB" ,"db_upload_queue.csv")
DB_PATH = os.path.join(pipeline_dir, "metadata", "DB" ,"db.csv")
DB_BACKUPS_PATH = os.path.join(pipeline_dir, "metadata", "DB" ,"db_backups")
## pipeline dirs and scripts ###
PIPELINE_LOGS_DIR = os.path.join(pipeline_dir, "metadata", "logs")
PIPELINE_RUNNER_SCRIPT = 'step_manager.py'

