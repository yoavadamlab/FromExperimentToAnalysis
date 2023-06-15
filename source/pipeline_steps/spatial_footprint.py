"""
based on: @caichangjia
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
import tifffile
import json

import caiman as cm
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    home_dir = gui_params[consts.HOME_DIR_LINUX]
    gui_time = gui_params[consts.GUI_TIME]
    input_video_flag = gui_params[consts.VOLPY_INPUT]
    input_video = get_input_video_path(raw_video_path, input_video_flag, gui_time)
    fr = pipe_utils.get_frame_rate(raw_video_path)
    volpy_dict = extract_volpy_params(gui_params, fr)
    return home_dir, raw_video_path, input_video, gui_time, volpy_dict

def extract_volpy_params(gui_params, fr):   
    volpy_params = {consts.CONTEXT_SIZE, consts.CENSOR_SIZE, consts.NPC_BG, consts.HP_FREQ_PB, consts.CLIP,
                    consts.MIN_SPIKES, consts.PNORM, consts.THRESHOLD, consts.RIDGE_BG, consts.SUB_FREQ, 
                    consts.N_ITER, consts.THRESHOLD_METHOD, consts.WEIGHT_UPDATE, consts.FLIP_SIGNAL}
    volpy_dict = {key: gui_params[key] for key in gui_params.keys() & volpy_params}
    for key in volpy_dict.keys():
        if key in [consts.CONTEXT_SIZE, consts.CENSOR_SIZE, consts.NPC_BG, consts.CLIP, consts.MIN_SPIKES, consts.THRESHOLD, consts.SUB_FREQ, consts.N_ITER]:
            volpy_dict[key] = int(volpy_dict[key])
        if key in [consts.HP_FREQ_PB, consts.PNORM, consts.RIDGE_BG]:
            volpy_dict[key] = float(volpy_dict[key])
    volpy_dict[consts.THRESHOLD_METHOD] = volpy_dict[consts.THRESHOLD_METHOD] 
    volpy_dict[consts.WEIGHT_UPDATE] = volpy_dict[consts.WEIGHT_UPDATE]
    volpy_dict[consts.FRAME_RATE] = fr
    volpy_dict[consts.VISUALIZE_ROI] = False
    volpy_dict[consts.DO_PLOT] = False
    return volpy_dict

def get_input_video_path(raw_video_path, input_video_flag, gui_time):
    if input_video_flag == consts.MC_INPUT:
        input_video = pipe_utils.get_mc_video(raw_video_path)
    elif input_video_flag == consts.DENOISED_INPUT:
        input_video = pipe_utils.get_denoised_path(raw_video_path, gui_time)
    return input_video

def run_volpy(input_video, rois, volpy_dict):
    def cluster_setup():
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)    
        return dview, n_processes  
    dview, n_processes =  cluster_setup()
    opts_dict = volpy_dict
    opts_dict["fnames"] = input_video
    opts_dict["index"] = list(range(len(rois)))# index of neurons
    opts_dict["ROIs"] = rois
    opts_dict["weights"] = None # if None, use ROIs for initialization
    opts = volparams(params_dict=opts_dict)
    #I add try and except for handling volpy input in tif format (rather mmap files)
    # thee added lines is in this file:
    # \\ems.elsc.huji.ac.il\adam-lab\qixin.yang\anaconda3\envs\caiman\lib\python3.6\site-packages\caiman\source_extraction\volpy\spikepursuit.py
    # between line 185 - 195
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
    cm.stop_server(dview=dview)
    dview.terminate()
    return vpy

def get_gui_time_dir(home_dir, gui_time):
    volpy_dir = os.path.join(home_dir, consts.PIPELINE_DIR, consts.VOLPY_DIR)
    pipe_utils.mkdir(volpy_dir)
    gui_time_dir = os.path.join(volpy_dir, gui_time)
    pipe_utils.mkdir(gui_time_dir)
    return gui_time_dir

def save_volpy_data(home_dir, gui_time_dir, vpy, rois, gui_time):
    np.save(os.path.join(gui_time_dir, consts.SLM_PATTERNS), rois)
    np.save(os.path.join(gui_time_dir, consts.VOLPY_DATA), vpy.estimates)
    np.save(os.path.join(gui_time_dir, consts.GUI_TIME), np.array([gui_time]))
    save_volpy_plots(home_dir, gui_time_dir, vpy.estimates)
    return 

def save_volpy_plots(home_dir, gui_time_dir, estimates):
    """
    volpy introduce some summary plots.
    we usally don't ise them but we save them anyway by this function
    """
    mean_img_path = os.path.join(home_dir, consts.PIPELINE_DIR, consts.MC_DIR, consts.MEAN_IMAGE + '.npy')
    mean_img = np.load(mean_img_path, allow_pickle=True)
    good_roi = np.where(estimates['locality'])[0]
    print("Neurons (as ROIs numbers) that pass locality test:\n", good_roi)
    for i in good_roi:
        fig = plt.figure(figsize=(10, 10))
        axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
        ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
        ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
        ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
        frame_times = np.array(range(len(estimates['t'][0])))
        vmax = np.percentile(mean_img, 99)
        ax1.cla()
        imgtmp = estimates['weights'][i]
        ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
        ax1.set_title(f'Spatial component {i+1}')
        ax1.axis('off')
        ax2.cla()
        ax2.plot(frame_times, estimates['t'][i], alpha=0.8)
        ax2.plot(frame_times, estimates['t_sub'][i])            
        ax2.plot(frame_times, estimates['t_rec'][i], alpha = 0.4, color='red')
        ax2.plot(frame_times[estimates['spikes'][i]],
                 1.05 * np.max(estimates['t'][i]) * np.ones(estimates['spikes'][i].shape),
                 color='r', marker='.', fillstyle='none', linestyle='none')
        ax3.cla()
        ax3.imshow(mean_img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
        imgtmp2 = imgtmp.copy()
        imgtmp2[imgtmp2 == 0] = np.nan
        ax3.imshow(imgtmp2, interpolation='None', alpha=0.2, cmap=plt.cm.hot)
        ax3.axis('off')
        plt.savefig(os.path.join(gui_time_dir, str(i) +'_roi_data.png'))
        pickle.dump(fig, open(os.path.join(gui_time_dir, str(i) +'_FigureObject.fig.pickle'), 'wb')) 
        plt.close()
    return

def calculate_traces(home_dir, raw_video_path, rois, vpy, gui_time):
    """
    apply the calculated spatial footprint from volpy on each cell 
    in the motion corrected video and extract the traces.
    """
    mc_path = pipe_utils.get_mc_video(raw_video_path)
    mc_video = tifffile.imread(mc_path)
    weights = vpy.estimates['weights']
    traces = pipe_utils.trace_extraction(mc_video, rois, weights)
    svae_weighted_traces(traces, home_dir, gui_time)
    return

def svae_weighted_traces(traces, home_dir, gui_time):
    pipeline_dir = os.path.join(home_dir, consts.PIPELINE_DIR)
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    weighted_traces_dir = os.path.join(mc_dir, consts.WEIGHTED_TRACES_DIR)
    pipe_utils.mkdir(weighted_traces_dir)
    num_of_cells = len(traces.columns)
    cols_names = [consts.SPATIAL_FOOTPRINT_TRACES_PREFIX + str(i+1) for i in range(num_of_cells)]
    traces.columns = cols_names
    path = os.path.join(weighted_traces_dir, gui_time + consts.TRACES_PATH)
    traces.to_csv(path, index=False, header=False)    
    return

def main(args):
    parameters_path = args[1]
    home_dir, raw_video_path, input_video, gui_time, volpy_dict = extract_params(parameters_path)
    rois = pipe_utils.get_rois_mask(raw_video_path)
    gui_time_dir = get_gui_time_dir(home_dir, gui_time)
    print ("GUI TIME:", gui_time)
    print("Calculate Spatial component on:", input_video)
    volpy_object = run_volpy(input_video, rois, volpy_dict)
    save_volpy_data(home_dir, gui_time_dir, volpy_object, rois, gui_time)
    calculate_traces(home_dir, raw_video_path, rois, volpy_object, gui_time)
    print(consts.STEP_COMPLETED) 
    return 

if __name__ == "__main__":
    main(sys.argv)