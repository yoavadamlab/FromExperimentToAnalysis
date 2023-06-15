import streamlit as st 
import numpy as np
import pandas as pd 
import os
from matplotlib.path import Path
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal
from scipy import stats    
import cv2
import ast 
import json
import spike_detection_utils as spike_utils
import data_utils as data_utils
import pipeline_utils as pipe_utils

################### data functions #########################

DATA_DIR = "Z:\Adam-Lab-Shared\Data\Behavior_and_Imagging"
TRACE_COL = "spatial_component_on_mc_cell_"
SPIKE_COL = "spikes_cell_"

def get_metadata():
    tmp_file_for_spike_detection = r"Z:\Adam-Lab-Shared\Data\Behavior_and_Imagging\spike_detection_temp.json"
    with open (tmp_file_for_spike_detection, 'rb') as fp:
        new_line = json.load(fp)
    cage_name, mouse_name, pipeline_seq = new_line[1:4]
    good_cells = new_line[4]
    cell_type = new_line[5]
    fov = new_line[6]
    fr = new_line[7]
    return cage_name, mouse_name, pipeline_seq, good_cells, cell_type, fov, fr

def get_mouse_df(cage_name, mouse_name, pipeline_seq):
    data_file = os.path.join(DATA_DIR, cage_name, mouse_name, pipeline_seq + '.parquet')
    df = pd.read_parquet(data_file)
    return df

def get_all_cells_traces(df, good_cells):
    all_traces = {}
    for cell in good_cells:
        trace = get_cell_trace(df, cell)
        all_traces[cell] = trace
    return all_traces

def get_cell_trace(df, cell):
    trace_prefix = data_utils.get_trace_col_prefix(df)
    trace = df[trace_prefix + str(cell)]
    trace = data_utils.detrend_func(np.arange(len(trace)), trace)
    return trace 

############### end of data functions ###########################

############### analysis functions ###################################

def explore_threshold(trace, fr, detection_method, segments_num, hyper_param_lst):
    segments = np.array_split(trace, segments_num)
    segments_size = [seg.size for seg in segments]
    if detection_method == "Volpy":
        detection_method_func = spike_utils.volpy_spike_detection
    if detection_method == "Median filter":
        detection_method_func = spike_utils.median_filter_detection
    spike_time_segments = []
    hp_segments = []
    ths = []
    time_stamp = 0
    for i, segment in enumerate(segments):
        spikes, hp_segment, th = detection_method_func(segment, fr, hyper_param_lst[i])
        spike_time_segments.append(spikes + time_stamp)
        hp_segments.append(hp_segment)
        ths.append(th)
        time_stamp += segments[i].size
    spike_time = np.hstack(spike_time_segments)
    hp_trace = np.hstack(hp_segments)
    return trace, hp_trace, spike_time, ths, segments_size

def get_exploration_figure(cage_name, mouse_name, pipeline_seq, cell, cell_type, trace, hp_trace, spike_time, ths, segments_size, spike_time_from_file):
    trace_color = 'red' if cell_type == 'Pyr' else 'blue'
    spike_color = 'blue' if cell_type == 'Pyr' else 'red'
    spike_value = 1.01 * trace.iloc[spike_time]
    file_spike_value = 1.04 * trace.iloc[spike_time_from_file]
    trace_ax_low, trace_ax_lim = get_secondary_axis_limit(trace)
    hp_ax_low, hp_ax_lim = get_axis_limit(hp_trace)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], shared_xaxes=True,  y_title='', x_title="Frame [#]")
    fig.add_scatter(name='trace', x=np.arange(len(trace)), y=trace,
                    line=dict(color=trace_color, width=0.8), showlegend=True, row=1, col=1, secondary_y=True)
    fig.add_scatter(name='high_pass', x=np.arange(len(hp_trace)), y=hp_trace, secondary_y=False,
                    line=dict(color="green", width=0.8), showlegend=True, row=1, col=1)
    fig.add_scatter(name='cell '+str(cell) + " spikes", x=spike_time, y=spike_value, mode='markers', secondary_y=True,
                    marker=dict(size=3, color=spike_color), visible='legendonly', row=1, col=1, showlegend=True),
    fig.add_scatter(name='cell '+str(cell) + " spikes in file", x=spike_time_from_file, y=file_spike_value, mode='markers', secondary_y=True,
                    marker=dict(size=3, color='purple'), visible = True, row=1, col=1, showlegend=True)
    fig.update_layout(title=cage_name + " | " + mouse_name + " | seq: " + pipeline_seq + " | cell: " + str(cell), title_x=0.5, font=dict(size=12,))
    fig.update_yaxes(title_text = "", row=1, col=1, range=[trace_ax_low, trace_ax_lim], secondary_y=True, showticklabels=True)
    fig.update_yaxes(title_text = "", row=1, col=1, range=[hp_ax_low, hp_ax_lim], secondary_y=False, showticklabels=True)

    # threshold plot
    colors = ['red', 'orange', 'yellow', 'black', 'indigo', 'purple', 'pink', 'brown', 'gray',"blue"]
    colors = colors[:len(ths)]
    last_x_point = 0 
    for segments_size, th, color in zip(segments_size, ths, colors):
        fig.add_shape(type='line',
                    x0 = last_x_point,
                    y0 = th,
                    x1 = last_x_point + segments_size,
                    y1 = th,
                    line=dict(color=color))
        last_x_point += segments_size
    return fig

def get_axis_limit(trace):
    trace_ax_lim = trace.max() + 0.25 * (trace.max() - trace.min())
    trace_ax_low = trace.min() - 0.8 * (trace.max() - trace.min())
    return trace_ax_low, trace_ax_lim

def get_secondary_axis_limit(trace):
    trace_ax_lim = 1.75 * (trace.max() - trace.min())
    trace_ax_low = trace.min() - 0.25 * (trace.max() - trace.min())
    return trace_ax_low, trace_ax_lim


############### Display functions ###################################

def streamlit_config(cage, mouse_name, pipeline_seq):
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: grey;'>Adam Lab</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Spike detection</h2>",
                unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Mouse: {} | {} </h4>".format(cage, mouse_name),
                unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>Pipeline Seq: {}<br><br></h5>".format(pipeline_seq),
                unsafe_allow_html=True)
    st.empty()
    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #008000;
            color:#ffffff;
        }
        div.stButton > button:hover {
            background-color: #006400;
            color:#ffffff;
            }

        </style>""", unsafe_allow_html=True)
    return

def sliders_display(good_cells, cell_type):
    default_method = 0 if cell_type == 'IN' else 1
    col1, col2, col3 = st.columns(3)
    with col1:
        cell = st.radio('Select cell: ', good_cells, key = 'cell_selector')
    with col2:
        detection_methods = ['Volpy', 'Median filter']
        detection_method = st.radio("Detection method:", detection_methods, key = cell, index = default_method)
    with col3:
        seg_num = 2 if cell_type == 'IN' else 1
        segments_num = st.select_slider('Select trace partitioning amount', options = np.arange(1,11), key = 'segments_slider_celll_' + str(cell), value = seg_num)
    parameters_sliders_lst = []
    for i, col in enumerate(st.columns(int(segments_num))):
        if detection_method == detection_methods[0]:
            help_text = "large p reasult with lower threshold and vice versa" if i == 0 else ''
            slider_text = 'Select p value for segnemt # '+str(i+1) if segments_num > 1 else 'Select p value'
            slider_range = np.round(np.linspace(0,1,21)[1:],2)
            slider_key = '_p_slider_'
            slider_initial_value = 0.5
        if detection_method == detection_methods[1]:
            help_text = "large sigma reasult with higher threshold and vice versa" if i == 0 else ''
            slider_text = 'Select std multiplicand for segnemt # '+str(i+1) if segments_num > 1 else 'Select std multiplicand'
            slider_range = np.round(np.linspace(0,10,21)[1:],2)
            slider_key = '_std_slider_'
            slider_initial_value = 4.5
        with col:
            slider_initial_value = 0.5 if cell_type == 'IN' else 4.5
            slider = st.select_slider(slider_text, options = slider_range, help = help_text, key =str(cell) + slider_key + str(i), value = slider_initial_value)    
        parameters_sliders_lst.append(slider)
    return parameters_sliders_lst, detection_method, segments_num, cell

def run_spike_detection(cage_name, mouse_name, pipeline_seq, good_cells, cell_type, fov, fr, df):
    parameters_sliders_lst, detection_method, segments_num, cell = sliders_display(good_cells, cell_type)
    trace = get_cell_trace(df, cell)
    spike_time_from_file = df[df[SPIKE_COL+str(cell)]==1].index 
    trace, hp_trace, spike_time, ths, segments_size = explore_threshold(trace, fr, detection_method, segments_num, parameters_sliders_lst)
    fig = get_exploration_figure(cage_name, mouse_name, pipeline_seq, cell, cell_type, trace, hp_trace, spike_time, ths, segments_size, spike_time_from_file)

    def save_spike_vector(df, spike_time, cage_name, mouse_name, pipeline_seq, cell, trace):
        spike_train = np.zeros(trace.size)
        spike_train[spike_time] = 1
        df[SPIKE_COL+str(cell)] = spike_train
        data_file = os.path.join(DATA_DIR, cage_name, mouse_name, pipeline_seq + '.parquet')
        df.to_parquet(data_file, index=False)
        return
    
    c1,c2,c3,c4,c5 = st.columns(5)
    for c in [c1,c2,c4,c5]:
        with c:
            pass
    with c3:
        save_spike_train = st.button('Save spikes', key = "save_button_" + str(cell))
    if save_spike_train:
        save_spike_vector(df, spike_time, cage_name, mouse_name, pipeline_seq, cell, trace)
    return fig

