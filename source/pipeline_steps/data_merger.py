import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import string
import json
import sys
import glob
import shutil
import xmltodict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils
from utils import files_paths as pipe_paths


def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    home_dir = gui_params[consts.HOME_DIR_LINUX]
    gui_time = gui_params[consts.GUI_TIME]
    cage = gui_params[consts.CAGE]
    mouse_name = gui_params[consts.MOUSE_NAME] 
    imaging_seq = gui_params[consts.SEQ] 
    return home_dir, gui_time, cage, mouse_name, imaging_seq

def find_pipeline_traces(home_dir):
    """
    Find which of the follwing steps was calculated during the pipline:
    1. Raw video traces
    2. Motion corrected traces
    3. Denoised traces # Not implemented! if want to add this go to get_traces_paths function
    4. Spatial footprint applied on the motion corrected video traces
    5. volpy outputs [traces of different intermidate steps + spikes columns]
    return pandas dataframe with N * num_of_cells columns, 
    while N equal to the number of steps were found. 
    """
    traces_paths, pipeline_dir = get_traces_paths(home_dir)
    traces_prefixes = [consts.RAW_TRACES_PREFIX, consts.MC_TRACES_PREFIX, consts.DENOISED_TRACES_PREFIX, consts.SPATIAL_FOOTPRINT_TRACES_PREFIX]
    df_lst = []
    print("The following pipeline outputs were found for merge:")
    for path, prefix in zip(traces_paths, traces_prefixes):
        if os.path.isfile(path):
            traces_df = pd.read_csv(path, header=None)  
            traces_df = traces_df.set_axis([prefix +  str(i+1) for i in range(len(traces_df.columns))], axis=1)
            df_lst.append(traces_df)
    vpy_path = get_volpy_dir_path(pipeline_dir)
    if os.path.exists(vpy_path):
        volpy_df = parse_volpy_data(vpy_path)
        df_lst.append(volpy_df)
    pipeline_outputs = pd.concat(df_lst, axis=1)
    return pipeline_outputs

def get_traces_paths(home_dir):
    """
    search all the paths for the traces that been calculated during the pieline.
    assume strict directories structere.
    """
    # frist decide what the name of the tarces dir, since we changed it on March 2023
    trace_dir_name = consts.TRACES_DIR
    pipline_dir_name = consts.PIPELINE_DIR 
    spatial_footprint_dir_name = consts.WEIGHTED_TRACES_DIR
    trace_suffix = consts.TRACES_PATH
    if not os.path.exists(os.path.join(home_dir, trace_dir_name)):
        trace_dir_name = consts.INTENSITIES_DIR
        pipline_dir_name = consts.OLD_VOLPY_DIR
        spatial_footprint_dir_name = consts.OLD_SC_DIR
        trace_suffix = consts.INTENS_PATH
    # now create variables for each reaces dir
    raw_traces_dir = os.path.join(home_dir, trace_dir_name)
    pipeline_dir = os.path.join(home_dir, pipline_dir_name)
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    mc_traces_dir = os.path.join(mc_dir, trace_dir_name)
    sf_on_mc_traces_dir = os.path.join(mc_dir, spatial_footprint_dir_name)
    denoised_dir = "" # don't delete - place holder for the right ordering of the list
    # noe grab the traces file path itself
    traces_paths = []
    for dirname in [raw_traces_dir, mc_traces_dir, denoised_dir, sf_on_mc_traces_dir]:
        if os.path.exists(dirname):
            path = pipe_utils.get_last_modified_file(dirname, trace_suffix)
        else:
            path = ""
        traces_paths.append(path)
    return traces_paths, pipeline_dir

def get_volpy_dir_path(pipeline_dir):
    volpy_dir = os.path.join(pipeline_dir, consts.VOLPY_DIR)
    vpy_name = consts.VOLPY_DATA + '.npy'
    if not os.path.exists(volpy_dir):
        volpy_dir = os.path.join(pipeline_dir, consts.OLD_VOLPY_DEMIX)
        vpy_name = consts.OLD_VPY_NAME + '.npy'
    if not os.path.exists(volpy_dir):
        return ""
    list_of_dirs = glob.glob(volpy_dir + "/*/") 
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    vpy = os.path.join(latest_dir, vpy_name)
    return vpy

def parse_volpy_data(volpt_npy_path) :
    npy = np.load(volpt_npy_path, allow_pickle = True).item()
    fields = ["F0", "t_rec", "t_sub", "t", "ts"]
    # npy["F0"] - baselinesignal
    # npy["t_rec"] - reconstructed spikes! zero for the threshhold - extract spike from this
    # npy["t_sub"] - sub threshold activity  i.e without spikes
    # npy["t"] - the signal after the pipeline, without whitened 
    # npy["ts"] - the signal after the pipeline, with whitened method
    # spikes genearted manually from t_rec
    df_lst = []
    for i in fields:
        df = pd.DataFrame(npy[i]).T
        df.columns = ["demix_volpy_" + i + '_cell_' + str(j + 1) for j in range(len(df.columns)) ]
        df_lst.append(df)
    demix_shape = npy["F0"].shape
    spikes = np.zeros(demix_shape)
    for cell,spike_timimg in enumerate(npy["spikes"]):
        for spike in spike_timimg:
            spikes[cell][spike] = 1
    df = pd.DataFrame(spikes).T
    df.columns = ["demix_volpy_spikes_cell_" + str(j + 1) for j in range(len(df.columns)) ]
    df_lst.append(df)
    volpy_data = pd.concat(df_lst, axis=1)
    return volpy_data

def create_behavioral_df(home_dir, cage, mouse_name, imaging_seq):
    virmen_path = os.path.join(pipe_paths.VIRMEN_DIR, cage, mouse_name, consts.VIRMEN_PREFIX + imaging_seq + '.csv')
    thorsync_path = get_thorsync_path(home_dir) 
    if (not os.path.exists(virmen_path)) or (not os.path.exists(virmen_path)):
        print("merge only pipeline steps, behavioral data wasn't found")
        return None
    vir = get_virmen_df(virmen_path)
    thor = get_thorsync_data(thorsync_path)
    behavioral_df = align_behavior_with_frames(vir, thor)
    return behavioral_df

def get_virmen_df(path):
    virmen_data = pd.read_csv(path)
    virmen_data[consts.VIR_TIME] = virmen_data[consts.VIR_TIME] * 1000
    return virmen_data

def get_thorsync_path(home_dir): 
    list_of_dirs = glob.glob(home_dir + "/{}*/".format(consts.TS_DIR_PREFIX) ) # * means all if need specific format then *.csv
    if len(list_of_dirs) == 0:
        return ""
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    h5_files = glob.glob(latest_dir + '/*.h5')
    if len(h5_files) == 0:   
        return ""
    return h5_files[0]

def get_thorsync_data(thor_path):
    df = extract_TS_data(thor_path)
    ts_frame_duration = get_ts_sample_rate(thor_path)
    if consts.TS_REWARD in df.columns:
        df[consts.TS_REWARD_COUNTER] = np.where(df[consts.TS_REWARD].astype(float).diff() < 0, 1,0).cumsum()
    df[consts.TS_CAMERA_FRAME_NUMBER] = np.where(df[consts.TS_FRAMEOUT].astype(float).diff() < 0, 1,0).cumsum()
    df[consts.TS_CAMERA_NEW_FRAME] = df[consts.TS_CAMERA_FRAME_NUMBER].diff()
    df["DAQ_Trigger_cumsum"] = df[consts.TS_TRIGGER].cumsum()
    df = df[df["DAQ_Trigger_cumsum"] > 0]
    df = df.drop('DAQ_Trigger_cumsum', axis=1)
    df = df.drop(consts.TS_TRIGGER, axis=1)
    df[consts.TS_TIME] = np.arange(1,len(df)+1) * ts_frame_duration
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def extract_TS_data(thor_path):
    df_cols = []
    keep_ts_cols =  [consts.TS_FRAMEOUT, consts.TS_BLUE_LASER, consts.TS_TRIGGER,
                                consts.TS_FRAME_COUNTER, consts.TS_REWARD, consts.TS_LICK]
    f = h5py.File(thor_path, 'r')
    for key in f.keys():
        group = f[key]
        for key in group.keys():
            if key in keep_ts_cols:
                data = group[key][()].squeeze()
                column = pd.DataFrame({key: data})
                column.reset_index(drop=True, inplace=True)
                df_cols.append(column)
    df = pd.concat(df_cols, axis = 1)
    return df

def get_ts_sample_rate(thor_path):
    ts_xml = os.path.join(os.path.dirname(thor_path), consts.TS_XML)
    xml_data = open(ts_xml, "r").readlines()[1:]
    xml_data = ' '.join(xml_data)
    xml_dict = xmltodict.parse(xml_data)
    boards =xml_dict["RealTimeDataSettings"]["DaqDevices"]["AcquireBoard"]
    for board in boards:
        if board["@active"] == '1':
            active_board = board
            break
    sample_rates = active_board["SampleRate"]
    for sample_rate in sample_rates:
        if sample_rate["@enable"] == '1':
            active_sample_rate = sample_rate
            break
    sample_rate = int(active_sample_rate['@rate'])
    frame_duration_ts_ms = (1/sample_rate) * 1000
    return frame_duration_ts_ms

def align_behavior_with_frames(vir, thor):
    df = pd.merge_asof(thor, vir, left_on=consts.TS_TIME, right_on=consts.VIR_TIME)
    df = df.fillna(method="bfill")
    df = df[np.where(df[consts.TS_FRAME_COUNTER].astype(float).diff() > 0, 1,0)==1]
    return df

def merge_imaging_and_behavior(imaging_data, behavioral_data):
    if behavioral_data is None:
        return imaging_data
    num_of_frames = len(imaging_data)
    behavioral_data = behavioral_data.iloc[:num_of_frames, :]
    behavioral_data.reset_index(drop=True, inplace=True)
    merged_df = pd.concat([behavioral_data, imaging_data], axis = 1)
    return merged_df

def extract_parquet_df(df):
    """
    extraction of only valuable columns to keep the mergede files small to further fast analysis
    """
    df = df.copy()
    informative_cols = [consts.TS_TIME, consts.TS_FRAME_COUNTER, consts.VIR_TIME, consts.SPEED, consts.POSITION,
       consts.LAP_LEN_CUMSUM, consts.VIR_LICK, consts.VIR_REWARD, consts.LAP_COUNTER, consts.MOVEMENT, consts.WORLD]
    all_cols = df.columns
    cols_to_keep = []
    for col in all_cols:
        if (col in informative_cols) or (col.startswith(consts.VOLPY_SPIKES)) or (col.startswith(consts.SPATIAL_FOOTPRINT_TRACES_PREFIX)):
            cols_to_keep.append(col)
    df = df[cols_to_keep]
    spikes_cols = [i for i in df.columns if i.startswith(consts.VOLPY_SPIKES) ]
    cols_to_rename = {old_col: consts.SPIKES + str(i+1) for i ,old_col in enumerate(spikes_cols)}
    df.rename(columns=cols_to_rename, inplace=True)
    return df

def validate_merge(df):
    if consts.VIR_REWARD not in df.columns:
        return # behavioral data wasn't found
    if consts.TS_REWARD_COUNTER not in df.columns:
        print("reward ttl from ThorSync doesn't present in the TS file")
        return
    df["reward_counter_vir"] = np.where(df[consts.VIR_REWARD].astype(float).diff() > 0, 1,0).cumsum()
    vir_frames = df.groupby('reward_counter_vir')[consts.TS_FRAME_COUNTER].count().to_numpy()
    TS_frames = df.groupby(consts.TS_REWARD_COUNTER)[consts.TS_FRAME_COUNTER].count().to_numpy() 
    if len(vir_frames) != len(TS_frames):
        print("different num of rewards between TS and VIRMEN")
    else:
        mean_diff = (vir_frames - TS_frames).mean()
        print("Behavior and Imaging alignment test:")
        print("mean differnce between frames per reward number =", mean_diff)
        print("Difference per reward number =", (vir_frames - TS_frames))

def save_data(df, home_dir, cage, mouse_name, imaging_seq):
    """
    The resulting files will be saved in a dedicated directory.
    they will sorted in folders per mouse.
    2 nain files will be saved - a csv file containing all the steps of the pipeline
    and a parquet file with kess columns, containig only the last step of the pipeline and the 
    usable columns from the behavioral table.
    alongside those files we are saving a few more files that will be used for visualization
     - the mean image and the weight matrix.
    the file names will be acoording to the imaging seq number. 
    if multiple pipeline were ruuned on the same umaging seq, a new file will be 
    saved with a letter after the seq number. the letters will be in an alphabetic order.
    """
    with open(os.path.join(home_dir, "imaging_session_" + str(imaging_seq) + '.txt'), 'w+') as f:
        f.write('Empty file, just indicate the imagging session')

    parquet_data = extract_parquet_df(df)
    cage_path = os.path.join(pipe_paths.DATASET_DIR, cage)
    mouse_path = os.path.join(cage_path, mouse_name)
    pipe_utils.mkdir(cage_path)
    pipe_utils.mkdir(mouse_path)

    file_suffix = get_files_suffix(mouse_path, imaging_seq)
    df.to_csv(os.path.join(mouse_path, file_suffix + '.csv'), index=False)
    parquet_data.to_parquet(os.path.join(mouse_path, file_suffix + '.parquet'), index=False)
    copy_pipeline_outputs_to_results_dir(mouse_path, home_dir, file_suffix)
    return 

def get_files_suffix(mouse_path, imaging_seq):
    suffixes = [""] + ["_" + l for l in list(string.ascii_lowercase)]
    for suf in suffixes:
        if not os.path.isfile(os.path.join(mouse_path, imaging_seq + suf + '.csv')):
            break # now suf = the right suffix for the file name
    return imaging_seq + suf

def copy_pipeline_outputs_to_results_dir(mouse_path, home_dir, file_suffix):
    file_suffix = '_' + file_suffix
    shifts_dir = os.path.join(home_dir, consts.PIPELINE_DIR, consts.MC_DIR, consts.SHIFTS_DIR)
    list_of_files = glob.glob(shifts_dir + '//*') 
    if len(list_of_files) > 0:
        latest_shift = max(list_of_files, key=os.path.getctime)
        shutil.copy(latest_shift, os.path.join(mouse_path, consts.MC_SHIFTS + file_suffix + '.mat')) 
    vpy_path = get_volpy_dir_path(os.path.join(home_dir, consts.PIPELINE_DIR))
    if os.path.exists(vpy_path):
        estimates = np.load(vpy_path, allow_pickle = True)
        np.save(os.path.join(mouse_path, consts.VOLPY_DATA + file_suffix), estimates)

        ROIs_before_demixing_path = os.path.join(os.path.split(vpy_path)[0], consts.SLM_PATTERNS + ".npy")
        ROIs_before_demixing = np.load(ROIs_before_demixing_path, allow_pickle=True)
        np.save(os.path.join(mouse_path, consts.SLM_PATTERNS + file_suffix), ROIs_before_demixing)

        mean_img_path = os.path.join(home_dir, consts.PIPELINE_DIR, consts.MC_DIR, consts.MEAN_IMAGE + ".npy")
        mean_img = np.load(mean_img_path, allow_pickle=True)
        np.save(os.path.join(mouse_path, consts.MEAN_IMAGE + file_suffix), mean_img)

    

def main(args):
    parameters_path = args[1]
    home_dir, gui_time, cage, mouse_name, imaging_seq = extract_params(parameters_path)
    print("GUI TIME:" , gui_time)
    imaging_data = find_pipeline_traces(home_dir)
    behavioral_data = create_behavioral_df(home_dir, cage, mouse_name, imaging_seq)
    df = merge_imaging_and_behavior(imaging_data, behavioral_data)
    validate_merge(df)
    save_data(df, home_dir, cage, mouse_name, imaging_seq) 
    print (consts.STEP_COMPLETED) 

if __name__ == "__main__":
     main(sys.argv)
    