### pipeline steps names ###
RAW_TRACES_EXTRACTION = 'raw_trace_extraction'
MOTION_CORRECTION = 'motion_correction'
SPATIAL_FOOTPRINT = 'spatial_footprint'
SPIKE_DETECTION = 'spike_detection'
DENOISING = 'denoising'
BEHAVIOR_AND_TRACES_MERGE = 'data_merger'
### pipeline states ###
WAITING = 0 
EXECUTING = 1
FINISHED = 2
FAILED = 3
CANCELLD = 4
### pipeline steps states ###
NOT_STARTED = "not started"
JOB_PENDING = "Pending"
JOB_RUNNING = "running"
JOB_FAILED = "failed"
JOB_FINISHED = "completed"
JOB_CANCELLD = "cancelled"
### SLURM states ###
SLURM_PENDING = "PENDING"
SLURM_RUNNING = "RUNNING"
SLURM_FAILED = "FAILED"
SLURM_FINISHED = "COMPLETED"
SLURM_CANCELLD = "CANCELLED"
### SLURM commands ###
RUN_JOB_COMMAND = "/opt/slurm/bin/sbatch"
JOB_STATE_COMMAND = "/opt/slurm/bin/sacct -j {} -n -o State -P"
CANCEL_JOB_COMMAND = "/opt/slurm/bin/scancel {}"
GET_LOG_PATH_COMMAND = "/opt/slurm/bin/scontrol show job {} | grep StdOut"
### directories and files names ###
PIPELINE_DIR = "pipeline_results"
VOLPY_DIR = "volpy_outputs"
RAW_VIDEO_TIF = "raw_video.tif"
MC_DIR = "motion_corrected"
SHIFTS_DIR = "shifts"
MEAN_IMAGE = 'mean_image'
MC_RAW_PATH = "_mc_and_raw.tif"
MC_VIDEO_PATH = "_motion_corrected.tif"
MC_SHIFTS_PATH = "_motion_corrected_shifts.mat"
MC_SHIFTS = "mc_shifts"
TRACES_DIR = "traces"
WEIGHTED_TRACES_DIR = "spatial_footprint_traces"
TRACES_PATH = "_traces.csv"
SLM_PATTERNS = "slm_patterns"
VOLPY_DATA = 'volpy_data'
VIRMEN_PREFIX = "imaging_"
TS_DIR_PREFIX = "TS"
REMOVED_LAPS = "_removed_laps"
TS_XML = "ThorRealTimeDataSettings.xml"
 ## old dirs and files ## 
INTENSITIES_DIR = "Intensities"
INTENS_PATH = "_intens.csv"
OLD_VOLPY_DIR = "volpy_results"
OLD_SC_DIR = "spatial_component_on_mc"
OLD_VOLPY_DEMIX = "volpy_demix"
OLD_VPY_NAME = "volpy_demix_data"
OLD_SLM_PATTERNS_NAME = "ROIs_before_demix"
### columns names ###
RAW_TRACES_PREFIX = "raw_cell_"
MC_TRACES_PREFIX = "mc_cell_"
DENOISED_TRACES_PREFIX = "denoised_cell_"
SPATIAL_FOOTPRINT_TRACES_PREFIX = "sf_on_mc_cell_"
OLD_SPATIAL_FOOTPRINT_TRACES_PREFIX = "spatial_component_on_mc_cell_"
VOLPY_SPIKES = "demix_volpy_spikes_cell_"
SPIKES = "spikes_cell_"
VIR_TIME = "time_from_trigger"
VIR_REWARD = "reward"
VIR_LICK = "lick"
SPEED = "speed"
POSITION = "position"
MOVEMENT = "movement"
LAP_COUNTER = "lap_counter"
WORLD = "current_World"
LAP_LEN_CUMSUM = "lap_length_cumsum"
TS_FRAMEOUT = "FrameOut"
TS_CAMERA_FRAME_NUMBER = "frame_number_from_camera"
TS_CAMERA_NEW_FRAME = "new_frame"
TS_BLUE_LASER = "BlueLaser"
TS_TRIGGER = "DAQ_Trigger"
TS_FRAME_COUNTER = "FrameCounter"
TS_REWARD = "reward_TS"
TS_LICK = "lick_TS"
TS_REWARD_COUNTER = "reward_counter_TS"
TS_TIME = "TS_time"
### GUI params ###
 ## general ##
CAGE = "cage"
MOUSE_NAME = "mouse_name"
SEQ = "seq"
CELL_TYPE = "cell_type"
GUI_TIME = "gui_time"
RAW_VIDEO_PATH = "raw_video_path"
RAW_VIDEO_PATH_LINUX = "raw_video_path_linux"
HOME_DIR_LINUX = "home_dir_linux"
HOME_DIR = "home_dir"
PARTIAL_VIDEO = "partial_video"
START_FRAME = "start_frmae_partial"
END_FRAME = "end_frmae_partial"
PARAMS_DIR_NAME = "pipeline_params"
PARAMS_FILE_SUFFIX_NAME = "_params.json"
 ## Motion correction ##
FRAME_RATE = "fr"
PW_RIGID = "pw_rigid"
GSIG_FILT = "gSig_filt"
MAX_SHIFTS = "max_shifts"
MAX_SHIFTS_X = "max_shifts_x"
MAX_SHIFTS_Y = "max_shifts_y"
STRIDES = "strides"
STRIDES_X = "strides_x"
STRIDES_Y = "strides_y"
OVERLAPS = "overlaps"
OVERLAPS_X = "overlaps_x"
OVERLAPS_Y = "overlaps_y"
MAX_DEVIATION_RIGID = "max_deviation_rigid"
 ## spatial component ##
VOLPY_INPUT = "volpy_input"
MC_INPUT = "mc_input"
DENOISED_INPUT = "denoised_input"
CONTEXT_SIZE = 'context_size'
CENSOR_SIZE = 'censor_size'
NPC_BG = 'nPC_bg'
HP_FREQ_PB = 'hp_freq_pb'
CLIP = 'clip'
MIN_SPIKES = 'min_spikes'
PNORM = 'pnorm'
THRESHOLD = 'threshold'
RIDGE_BG = 'ridge_bg'
SUB_FREQ = 'sub_freq'
N_ITER = 'n_iter'
THRESHOLD_METHOD = 'threshold_method'
WEIGHT_UPDATE = 'weight_update'
DO_PLOT = 'do_plot'
VISUALIZE_ROI = 'visualize_ROI'
FLIP_SIGNAL = 'flip_signal'
ADAPTIVE_THRESHOLD = "adaptive_threshold"
SIMPLE_THRESHOLD = "simple"
RIDGE = "ridge"
NMF = "NMF"
 ### Messages ####
STEP_COMPLETED = "Everything worked well. The Script finished to run."
### DB Schema ###

