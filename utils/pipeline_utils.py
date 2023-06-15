import os
import re
import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import xmltodict
from matplotlib.path import Path
import pandas as pd
import glob
import datetime
import sys
import scipy.io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.umask(0)
        os.makedirs(dir_name, mode=0o777)
    return 

def windows_to_linux_path(path):
    """
    Convert path on windoes OS to the corresponding cluster path.
    """
    pattern = "Adam-Lab-Shared"
    path_prefix = '/ems/elsc-labs/adam-y/Adam-Lab-Shared/'
    match = (re.search(pattern, path))
    path_suffix = path[match.end():].replace('\\','/')
    fixed_path = path_prefix + path_suffix
    return fixed_path

def get_raw_video_dimensions(raw_path):
    """
    extract video's width and height fro, experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab
    """
    xml = get_experiment_xml_path(raw_path)
    tree = ET.parse(xml)
    root = tree.getroot()
    width = int(root[5].attrib['width'])
    height = int(root[5].attrib['height'])
    return width, height

def get_experiment_xml_path(raw_path):
    """
    return the path of experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab.
    the function assumes that the path is in the same directory of the raw video
    """
    return os.path.join(os.path.split(raw_path)[0],'Experiment.xml')

def get_frame_rate(raw_path):
    xml_path = get_experiment_xml_path(raw_path)
    xml_data = open(xml_path,"r").read()
    xml_dict = xmltodict.parse(xml_data)
    exposure_time = float(xml_dict["ThorImageExperiment"]["Camera"]["@exposureTimeMS"])
    fr = np.round(1000/exposure_time).astype(int)
    return fr

def get_fov_name(raw_path):
    path_col_split = raw_path.split('\\')
    for word in path_col_split:
        if "FOV" in word:
            return word
    return np.nan

def get_pipeline_results_dir(raw_path):
    """
    return the path (and also create it if not exist) of the directory
    that will contain many outputs of the pipeline
    """
    pipeline_dir = os.path.join(os.path.split(raw_path)[0], consts.PIPELINE_DIR)
    mkdir(pipeline_dir)
    return pipeline_dir

def get_last_modified_file(dirname, suffix):
    list_of_files = glob.glob(dirname + '/*' + suffix) # * means all if need specific format then *.csv
    last_path = max(list_of_files, key=os.path.getctime) # take the last modified file
    return last_path

def get_mc_video(raw_video_path):
    pipeline_results = os.path.join(os.path.split(raw_video_path)[0], consts.PIPELINE_DIR) 
    if not os.path.exists(pipeline_results):
        pipeline_results = os.path.join(os.path.split(raw_video_path)[0], consts.OLD_VOLPY_DIR) # changed from consts.VOLPY_DIR
    mc_dir = os.path.join(pipeline_results, consts.MC_DIR)
    mc_path = get_last_modified_file(mc_dir, consts.MC_VIDEO_PATH)
    return mc_path   

def get_denoised_path(fnames, gui_time):
    home = os.path.split(os.path.split(fnames)[0])[0]
    denoise_dir = os.path.join(home,'denoiser_files')
    deepinterpolation_dir = os.path.join(denoise_dir,'deepinterpolation')
    deepvid_dir = os.path.join(denoise_dir,'deepvid')
    gui_time_dir = os.path.join(deepinterpolation_dir, gui_time)
    if not os.path.exists(gui_time_dir):
        gui_time_dir = os.path.join(deepvid_dir, gui_time)
    denoised_file = os.path.join(gui_time_dir, "denoised_no_pad.tif") 
    if os.path.isfile(denoised_file): # if merge done right after pipeline finished
        return denoised_file
    # else take the last file in denoise_dir 
    list_of_files = glob.glob(denoise_dir + '/**/*.tif', recursive=True) 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def raw_to_tif(raw_path, partial=None, start_frame=None, end_frame=None):
    """
    convert the raw formatted vodeo file to a tif file and save it in pipeline results dir.
    find width and height in the experiment.xml file.
    if start and end frame supplied - save sliced video based on those values.
    """    
    pipeline_dir = get_pipeline_results_dir(raw_path)
    width, height = get_raw_video_dimensions(raw_path)
    rawmovie_1d = np.fromfile(raw_path, dtype=np.uint16)
    movie_3d = np.reshape(rawmovie_1d,(-1,height,width)) 
    if partial:
        movie_3d = movie_3d[start_frame:end_frame,:,:]
    tif_video_path = os.path.join(pipeline_dir, consts.RAW_VIDEO_TIF)
    tifffile.imsave(tif_video_path, movie_3d, bigtiff=True)
    return tif_video_path

def get_rois_mask(raw_video_path):
    """
    for given raw video path, looking for the "ROIs.xaml" file from ThorImage
    and generate binary mask as np array in the sahpe of (#ROIs, height, width)
    """
    xml_path = os.path.join(os.path.split(raw_video_path)[0],'ROIs.xaml')
    xml_data = open(xml_path, "r").read()
    xml_dict = xmltodict.parse(xml_data)
    polygons_struct = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    # extract polygons of ROIS.
    poly_lst = []
    for i in range(len(polygons_struct)):
        p = polygons_struct[i]['@Points']
        if p not in poly_lst:
            poly_lst.append(p)
    # extract the coordinates of the rectangle ROI
    rect_data = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIRect"]
    bottom_left_x, bottom_left_y = [float(i) for i in rect_data["@BottomLeft"].split(',')]
    top_left_x, top_left_y = [float(i) for i in rect_data["@TopLeft"].split(',')]
    height = float(rect_data["@ROIHeight"])
    width = float(rect_data["@ROIWidth"])
    # generate list of polygons w.r.t the rectangle ROI
    corrected_polygons = []
    for polygon in poly_lst: # for each polygon
        corrected_points = []
        points = polygon.split(' ')
        for point in points:
            x, y = [float(i) for i in point.split(',')]
            # if the point exceeds the rectangle from above, left or right - trunc it
            x = min(max(x - bottom_left_x, 1),width) 
            y = max(1,min(y - top_left_y, height))
            corrected_points.append((x, y))
        corrected_points.append(corrected_points[0])
        corrected_polygons.append(corrected_points)
    # generate masks
    width, height = get_raw_video_dimensions(raw_video_path)
    ROIs = []
    for poly in corrected_polygons:
        flipped_poly = [(j,i) for i,j in poly]
        polygon = flipped_poly
        poly_path = Path(polygon)
        x, y = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(height, width)
        if mask.sum() > 0: # fot the case that a point was signed in the slm
            ROIs.append(mask)
    ROIs = np.stack(ROIs)
    return ROIs

def trace_extraction(video, rois_mask, weights=None):
    """
    video - 3d np array represent a video.
    rois - a binary np array in the shape of (#cells, width, height).
            its represent the pixels corresponding to each cell in the video.
    weights - represent spatial components to extract the traces accordingly.
            if not supplied - just preform non weighted mean over the cell
    """
    if weights is None:
        weights = rois_mask
    df_columns = ['cell ' + str(i+1) for i in range(len(rois_mask))]
    df = pd.DataFrame(columns=df_columns)
    for roi_num in range(len(rois_mask)):
        Xinds = np.where(np.any(rois_mask[roi_num] > 0, axis=1) > 0)[0]
        Yinds = np.where(np.any(rois_mask[roi_num] > 0, axis=0) > 0)[0]
        croped_video = video[:, Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        cell_mask = weights[roi_num]
        croped_mask = cell_mask[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        masked_video = croped_video * croped_mask[np.newaxis,:,:]
        trace = masked_video.mean(axis=(1, 2))
        df[df.columns[roi_num]] = trace
    return df

def get_video_details(path):
    """
    given raw video path, the function look after the cage, mouse name
    and imaging sequence 
    """
    try:
        cage = path.split("/")[4]
        name = path.split("/")[5]
        time_stamp = os.path.getmtime(path)
        video_time_object = datetime.datetime.fromtimestamp(time_stamp)
        virmen_data_dir = os.path.join(paths.VIRMEN_DIR_WINDOWS,cage,name)
        for file in glob.glob(virmen_data_dir + '\*.csv'):
            time_object = datetime.datetime.fromtimestamp(os.path.getmtime(file))
            file_date = time_object.date()
            file_hour = time_object.hour
            file_minute = time_object.minute
            if ((video_time_object.date() == file_date) and (video_time_object.hour == file_hour) and (video_time_object.minute == file_minute)):
                vir_file = file
                break
        imaging_session = os.path.basename(vir_file).split('_')[1].split('.')[0]
        return cage, name, imaging_session
    except:
        return None, None, None

def get_pipline_results_data(cage, mouse_name, seq):
    suffix = '_' + seq
    suffix = suffix.replace(consts.REMOVED_LAPS, '')
    volpy_data_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.VOLPY_DATA + suffix + '.npy')
    if not os.path.exists(volpy_data_path):
        suffix_old =  suffix.replace('_cut_laser_drop','')
        volpy_data_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.OLD_VPY_NAME + suffix_old + '.npy')
    volpy_data = np.load(volpy_data_path, allow_pickle=True).item()
    slm_patterns_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.SLM_PATTERNS + suffix + '.npy')
    if not os.path.exists(slm_patterns_path):
        suffix_old = '_' + seq.split('_')[0]
        slm_patterns_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.OLD_SLM_PATTERNS_NAME + suffix_old + '.npy')
    slm_patterns = np.load(slm_patterns_path, allow_pickle=True)
    mean_image_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.MEAN_IMAGE + suffix + '.npy')
    if not os.path.exists(mean_image_path):
        suffix_old = '_' + seq.split('_')[0]
        mean_image_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.MEAN_IMAGE + suffix_old + '.npy')
    mean_image = np.load(mean_image_path, allow_pickle=True)
    mc_shifts_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.MC_SHIFTS + suffix + '.mat')
    if not os.path.exists(mc_shifts_path):
        suffix_old =  suffix.replace('_cut_laser_drop','')
        mc_shifts_path = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, consts.MC_SHIFTS + suffix_old + '.mat')
    mc_shifts = scipy.io.loadmat(mc_shifts_path)
    mc_shifts = mc_shifts["motion_corrected_shifts"]
    return volpy_data, slm_patterns, mean_image, mc_shifts

def is_reemapping_experiment(df):
    try:
        n_worlds = df[consts.WORLD].nunique()
        if n_worlds == 2:
            return False
        elif n_worlds == 3:
            return True
        else:
            return -1
    except:
        return np.nan

def save_record_to_DB_queue(experiment_date, cage, mouse_name, seq, cell_type, video_path):
    df = pd.read_csv(paths.DB_UPLOAD_QUEUE_PATH)
    row = [experiment_date, cage, mouse_name, seq, cell_type, video_path]
    df.loc[len(df)] = row
    df.to_csv(paths.DB_UPLOAD_QUEUE_PATH, date_format='%Y-%m-%d', index=False)
    return 

def save_record_to_DB(row):
    df = pd.read_csv(paths.DB_PATH)
    df.loc[len(df)] = row
    df.to_csv(paths.DB_PATH, index=False)
    return 

def get_steps_lst():
    return [consts.RAW_TRACES_EXTRACTION, consts.MOTION_CORRECTION, 
                 consts.SPATIAL_FOOTPRINT, consts.BEHAVIOR_AND_TRACES_MERGE, consts.SPIKE_DETECTION]