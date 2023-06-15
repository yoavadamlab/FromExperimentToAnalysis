import numpy as np
import pandas as pd 
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import files_paths as paths
from utils import pipeline_constants as consts
from utils import data_utils as data_utils
from utils import pipeline_utils as pipe_utils
from utils import spike_detection_utils as spike_utils
import string
import warnings
warnings.filterwarnings('ignore')


def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    cage = gui_params[consts.CAGE]
    mouse_name = gui_params[consts.MOUSE_NAME] 
    imaging_seq = gui_params[consts.SEQ] 
    fr = pipe_utils.get_frame_rate(raw_video_path)
    cell_type = gui_params[consts.CELL_TYPE]
    return cage, mouse_name, imaging_seq, cell_type, fr

def get_mouse_df_path(cage_name, mouse_name, pipeline_seq):
    mouse_path = os.path.join(paths.DATASET_DIR, cage_name, mouse_name)
    file_suffix = get_files_suffix(mouse_path, pipeline_seq)
    df_path = os.path.join(os.path.join(mouse_path, file_suffix + '.parquet'))
    return df_path

def get_files_suffix(mouse_path, imaging_seq):
    suffixes = [""] + ["_" + l for l in list(string.ascii_lowercase)]
    i = 0
    for suf in suffixes:
        if not os.path.isfile(os.path.join(mouse_path, imaging_seq + suf + '.parquet')):
            break # now suf = the right suffix for the file name
        else:
            i += 1
    return imaging_seq + suffixes[i-1]

def get_cell_trace(df, cell):
    trace_col_prefix = data_utils.get_trace_col_prefix(df) 
    trace = df[trace_col_prefix + str(cell)]
    trace = data_utils.detrend_func(np.arange(len(trace)), trace)
    return trace

def detect_spikes(trace, fr, config):
    segments = np.array_split(trace, config['partitioning'])
    segments_size = [seg.size for seg in segments]
    if config['method'] == 'volpy':
        detection_method_func = spike_utils.volpy_spike_detection
        hyper_param = config['p']
    if config['method'] == 'median_filter':
        detection_method_func = spike_utils.median_filter_detection
        hyper_param = config['std']
    spike_time_segments = []
    hp_segments = []
    ths = []
    time_stamp = 0
    for i, segment in enumerate(segments):
        spikes, hp_segment, th = detection_method_func(segment, fr, hyper_param)
        spike_time_segments.append(spikes + time_stamp)
        hp_segments.append(hp_segment)
        ths.append(th)
        time_stamp += segments[i].size
    spike_time = np.hstack(spike_time_segments)
    hp_trace = np.hstack(hp_segments)
    spike_train = np.zeros(trace.size)
    spike_train[spike_time] = 1
    return spike_train

def get_detection_config(cell_type):
    conf = {}
    if cell_type == 'In':
        conf['method'] = 'volpy'
        conf['partitioning'] = 2
        conf['p'] = 0.5
    if cell_type == 'Pyr':
        conf['method'] = 'median_filter'
        conf['partitioning'] = 1
        conf['std'] = 4.5
    if cell_type == "Pyr & IN":
        conf['method'] = 'median_filter'
        conf['partitioning'] = 1
        conf['std'] = 4.5
    return conf

def main(args):
    parameters_path = args[1]
    cage, mouse_name, imaging_seq, cell_type, fr = extract_params(parameters_path)
    df_path = get_mouse_df_path(cage, mouse_name, imaging_seq)
    df = pd.read_parquet(df_path)
    spike_col_prefix = data_utils.get_spike_col_prefix(df)
    cells_numbers = data_utils.get_cells_list(df)
    config = get_detection_config(cell_type)
    for cell in cells_numbers:
        cell_trace = get_cell_trace(df, cell)
        df[spike_col_prefix + str(cell)] = detect_spikes(cell_trace, fr, config)
    df.to_parquet(df_path, index=False)
    return 

if __name__ == "__main__":
    main(sys.argv)