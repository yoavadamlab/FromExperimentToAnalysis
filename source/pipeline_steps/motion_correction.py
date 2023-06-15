"""
based on: @caichangjia
"""
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.io import savemat
import pandas as pd
import tifffile
import json
import sys
import numpy as np

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams
from caiman.summary_images import mean_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    partial_video = gui_params[consts.PARTIAL_VIDEO]
    start_frame, end_frame = int(gui_params[consts.START_FRAME]), int(gui_params[consts.END_FRAME])
    gui_time = gui_params[consts.GUI_TIME]
    fr = pipe_utils.get_frame_rate(video_path)
    mc_dict = extract_mc_params(gui_params, fr)
    return video_path, gui_time, mc_dict, partial_video, start_frame, end_frame

def extract_mc_params(gui_params, fr):   
    mc_params = {consts.PW_RIGID, consts.GSIG_FILT,
                consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y, 
                consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X, 
                consts.OVERLAPS_Y, consts.MAX_DEVIATION_RIGID}
    mc_dict = {key: gui_params[key] for key in gui_params.keys() & mc_params}
    for key in mc_dict.keys():
        if key != consts.PW_RIGID:
            mc_dict[key] = int(mc_dict[key])
    if mc_dict[consts.GSIG_FILT] == 0: # without high pass filtering
        mc_dict[consts.GSIG_FILT] = None
    else:
        mc_dict[consts.GSIG_FILT] = [mc_dict[consts.GSIG_FILT], 0] # 0 is just placeholder and doesnt really used in the mc code
    mc_dict[consts.STRIDES] = (mc_dict[consts.STRIDES_X], mc_dict[consts.STRIDES_Y])
    mc_dict[consts.OVERLAPS] = (mc_dict[consts.OVERLAPS_X], mc_dict[consts.OVERLAPS_Y])
    mc_dict[consts.MAX_SHIFTS] = (mc_dict[consts.MAX_SHIFTS_X], mc_dict[consts.MAX_SHIFTS_Y])
    to_delete = [consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y, consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X, 
                consts.OVERLAPS_Y,]
    for key in to_delete:
        del mc_dict[key]
    mc_dict[consts.FRAME_RATE] = fr
    return mc_dict

##### Caiman motion correction #####

def run_motion_correction(video_path, mc_params):
    def set_mc_parameters(video_path, mc_dict):
        opts_dict = mc_dict
        opts_dict["fnames"] = video_path
        opts_dict["border_nan"] = 'copy'
        opts = volparams(params_dict=opts_dict)
        return opts
    def cluster_setup():
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)    
        return dview  

    opts = set_mc_parameters(video_path, mc_params)
    dview = cluster_setup()
    
    try:	
        mc = MotionCorrect(video_path, dview=dview, **opts.get_group('motion'))	
        mc.motion_correct(save_movie=True)	
    except Exception as e:	
        print(str(e))	
        print("motion correction failed - trying to run another one with reduce shifts")	
        mc_params[consts.MAX_SHIFTS] = (10,10)	
        opts = volparams(params_dict=mc_params)	
        mc = MotionCorrect(video_path, dview=dview, **opts.get_group('motion'))	
        mc.motion_correct(save_movie=True)

    mean_img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)

    return mc, mean_img

##### saving functions #####
def save_mc_data(video_path_tif, pipeline_dir, gui_time, mc, mean_img):   
    raw_video = cm.load(video_path_tif)
    mc_video = cm.load(mc.mmap_file)
    ds_ratio = 0.1 # resize by taking frames every 1 * T (where T = num of frames in the video)
    mc_and_raw = cm.concatenate([raw_video.resize(1, 1, ds_ratio), mc_video.resize(1, 1, ds_ratio)], axis=2).astype('float16')
    
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    pipe_utils.mkdir(mc_dir)

    mc_raw_path = os.path.join(mc_dir, gui_time + consts.MC_RAW_PATH)
    tifffile.imsave(mc_raw_path, mc_and_raw, bigtiff=True)

    mc_video_path = os.path.join(mc_dir, gui_time + consts.MC_VIDEO_PATH)
    tifffile.imsave(mc_video_path, mc_video.astype('float16'),bigtiff=True)

    shifts_dir = os.path.join(mc_dir, consts.SHIFTS_DIR)
    pipe_utils.mkdir(shifts_dir)
    shifts_rig = {"motion_corrected_shifts": mc.shifts_rig}
    savemat(os.path.join(shifts_dir, gui_time + consts.MC_SHIFTS_PATH), shifts_rig)

    mean_img = np.array(mean_img.tolist())
    np.save(os.path.join(mc_dir, consts.MEAN_IMAGE), mean_img)

    os.remove(video_path_tif) # delete the raw.tif file after motion corrected (the .raw file stil exist of course)
    os.remove(mc.mmap_file[0]) # delete memory map file
    return mc_video


def svae_mc_traces(traces, pipeline_dir, gui_time):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    traces_dir = os.path.join(mc_dir, consts.TRACES_DIR)
    pipe_utils.mkdir(mc_dir)
    pipe_utils.mkdir(traces_dir)
    path = os.path.join(traces_dir, gui_time + consts.TRACES_PATH)
    traces.to_csv(path, index=False, header=False)    
    return

    
def main(args):
    gui_params_path = args[1]
    raw_video_path, gui_time, mc_params, partial, start_frame, end_frame = extract_params(gui_params_path)
    print ("GUI TIME:", gui_time)
    print("Motion Correcrion on:", raw_video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    video_path_tif = pipe_utils.raw_to_tif(raw_video_path, partial, start_frame, end_frame) 
    mc_object, mean_img = run_motion_correction(video_path_tif, mc_params)
    mc_video = save_mc_data(video_path_tif, pipeline_dir, gui_time, mc_object, mean_img)
    rois = pipe_utils.get_rois_mask(raw_video_path)
    traces = pipe_utils.trace_extraction(mc_video, rois)
    svae_mc_traces(traces, pipeline_dir, gui_time)
    print(consts.STEP_COMPLETED) 
    return 

if __name__ == "__main__":
     main(sys.argv)