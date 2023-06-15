import sys
import os
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except:
    pass
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths


def get_fov_data(cage, mouse_name, pipeline_seq):    
    data_file = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, pipeline_seq + '.csv')
    df = pd.read_csv(data_file)
    df = old_cols_names_conversion(df)
    return df

def get_parquet_data(cage, mouse_name, pipeline_seq):    
    data_file = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, pipeline_seq + '.parquet')
    df = pd.read_parquet(data_file)
    return df

def old_cols_names_conversion(df):
    # spike cols
    spikes_cols = [i for i in df.columns if i.startswith('demix_volpy_spikes_cell_') ]
    cols_to_rename = {old_col: consts.SPIKES + str(i+1) for i ,old_col in enumerate(spikes_cols)}
    df.rename(columns=cols_to_rename, inplace=True)
    # raw_traces
    raw_traces_cols = [i for i in df.columns if i.startswith('original_cell_') ]
    cols_to_rename = {old_col: consts.RAW_TRACES_PREFIX + str(i+1) for i ,old_col in enumerate(raw_traces_cols)}
    df.rename(columns=cols_to_rename, inplace=True)
    # Spatial footprint traces
    sf_traces_cols = [i for i in df.columns if i.startswith(consts.OLD_SPATIAL_FOOTPRINT_TRACES_PREFIX) ]
    cols_to_rename = {old_col: consts.SPATIAL_FOOTPRINT_TRACES_PREFIX + str(i+1) for i ,old_col in enumerate(sf_traces_cols)}
    df.rename(columns=cols_to_rename, inplace=True)
    if 'lick_y' in df.columns:
        df.rename(columns={'lick_y' :consts.VIR_LICK}, inplace=True)
    return df

def get_cells_list(df):
    trace_prefix =  get_trace_col_prefix(df)
    cols_names = df.columns
    cells_cols = [col for col in cols_names if col.startswith(trace_prefix)]
    cells =  list(range(1, len(cells_cols) + 1))
    return cells

def check_behavioral_data(df):
    cols_names = df.columns
    if consts.POSITION in cols_names:
        return True
    else: 
        return False

def get_spike_col_prefix(df):
    cols = df.columns
    if consts.SPIKES + '1' in cols:
        return consts.SPIKES
    else:
        return consts.VOLPY_SPIKES

def get_trace_col_prefix(df):
    cols = df.columns
    if consts.SPATIAL_FOOTPRINT_TRACES_PREFIX + '1' in cols:
        return consts.SPATIAL_FOOTPRINT_TRACES_PREFIX
    else:
        return consts.OLD_SPATIAL_FOOTPRINT_TRACES_PREFIX

def detrend_func(t, data):
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    popt, pcov = curve_fit(exp_func, t, data, p0=[920, 1/10000, 0], maxfev=10000)
    detrend_data = data - exp_func(t, *popt)
    return detrend_data

def remove_lpas(df, laps_to_remove):
    first_lap = df[consts.LAP_COUNTER].min()
    laps_to_remove = np.array(laps_to_remove)
    frame_interval = np.round(df[consts.TS_TIME].diff().mean(),2)
    df = df[~df.lap_counter.isin(first_lap + (laps_to_remove - 1))]
    df = df.reset_index(drop=True)
    df[consts.TS_TIME] = np.linspace(0, frame_interval *len(df), len(df), endpoint=False) # create new time column for visualization
    return df


def get_cell_contour(slm_patterns, cell_num):
    cell_contour = slm_patterns[cell_num-1]
    cell_contour.dtype = np.uint8
    cont_idx = cv2.findContours(
        cell_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cont_idx = max(cont_idx, key=len) # get the longest contour, since sometimes there are more than one in one patten
    cont_idx = cont_idx.reshape(-1, 2)
    return cont_idx

def fov_traces_figure(df, cells_list, behavior_flag, detrend, pipe_step):
    fig = get_layout_fov_traces_fig(cells_list, behavior_flag)
    step_col_prefix, trace_color, spike_color = pipeline_steps_featurs(pipe_step, df)
    fig = add_data_fov_traces_fig(fig, df, cells_list, behavior_flag, detrend, step_col_prefix, trace_color, spike_color)
    return fig 

def get_layout_fov_traces_fig(cells_lst, behavior_flag):
    num_of_cells = len(cells_lst)
    rows_num = num_of_cells
    row_heights = [25] * rows_num
    if behavior_flag:
        rows_num += 1
        row_heights += [5]
    fig = make_subplots \
        (
        rows = rows_num, cols = 1,
        vertical_spacing = 0.01, row_heights = row_heights, shared_xaxes=True, 
        row_titles = ["cell # " + str(i) for i in cells_lst],
        y_title = 'A.U', x_title = "Frames [#]"
        )
    if behavior_flag:
        fig['layout']['yaxis'+str(num_of_cells+1)]['title'] = 'Position'+'<br>'+'[cm]'
        fig['layout']['yaxis'+str(num_of_cells+1)]['title'].update(font=dict(size=10))
        fig.update_yaxes(showticklabels=False, row=num_of_cells+1, col=1)
    if num_of_cells > 1:
        fig.update_layout(height=num_of_cells*100, width=1000)
    else:
        fig.update_layout(height=400, width=1000)
    fig.update_layout(showlegend=True)
    return fig 

def add_data_fov_traces_fig(fig, df, cells_list, behavior_flag, detrend, step_col_prefix, trace_color, spike_color):
    for i, cell in enumerate(cells_list):
        trace = df[step_col_prefix + str(cell)]
        if detrend:
            trace = detrend_func(np.arange(len(trace)), trace)
        spike_col_prefix = get_spike_col_prefix(df)
        spike_times = df[df[spike_col_prefix + str(cell)] == 1].index
        spike_value = 1.01 * trace.iloc[spike_times]
        trace_ax_min, trace_ax_max = get_axis_limit(trace, 0.25, 0.25)
        fig.add_scatter(name=" spikes", x=spike_times, y=spike_value, mode='markers', legendgroup =2, showlegend=i==0,
                        marker=dict(size=2.5, color=spike_color), visible='legendonly', row=i+1, col=1)
        fig.add_scatter(name='cell '+str(cell), x=np.arange(len(trace)), y=trace,
                        line=dict(color=trace_color, width=0.4), showlegend=False, row=i+1, col=1)
        fig.update_yaxes(row=i+1, col=1, range=[trace_ax_min, trace_ax_max])

    if behavior_flag:
        num_of_cells = len(cells_list)
        behave_col = df[consts.POSITION]
        lick_times = df[df[consts.VIR_LICK] == 1].index
        lick_value = 1.01 * behave_col[lick_times]
        fig.add_trace(go.Scatter(x=np.arange(len(behave_col)), y=behave_col, yaxis="y2",
                    showlegend=False, marker=dict(color='black')), row=num_of_cells+1, col=1, )
        fig.add_scatter(name = " licks", x=lick_times, y=lick_value, mode='markers', marker=dict(
            size=2.5, color=spike_color), visible='legendonly', row=num_of_cells+1, col=1)
    return fig

def pipeline_steps_featurs(pipe_step, df=None):
    if pipe_step == consts.RAW_TRACES_EXTRACTION:
        step_col_prefix = consts.RAW_TRACES_PREFIX
        trace_color, spike_color = 'blue', 'red'
    if pipe_step == consts.MOTION_CORRECTION:
        step_col_prefix = consts.MC_TRACES_PREFIX
        trace_color, spike_color = 'magenta', 'red'
    if pipe_step == consts.SPATIAL_FOOTPRINT:
        if df is not None:
            step_col_prefix = get_trace_col_prefix(df)
        else:
            step_col_prefix = consts.SPATIAL_FOOTPRINT_TRACES_PREFIX
        trace_color, spike_color = 'green', 'red'
    return step_col_prefix, trace_color, spike_color

def get_axis_limit(trace, top_space, low_space):
    trace_ax_max = trace.max() + top_space * (trace.max() - trace.min())
    trace_ax_min = trace.min() - low_space * (trace.max() - trace.min())
    return trace_ax_min, trace_ax_max

def mc_shifts_fig(mc_shifts):
    fig = make_subplots \
        (
        rows = 2, cols = 1, shared_xaxes=True,
        vertical_spacing = 0,
        row_titles = ["horizontal <br> pixels shifts", "vertical <br> pixels shifts"],
        y_title = 'pixels', x_title = "Frames [#]"
        )
    
    fig.update_layout(height=400, width=1000)
    fig.add_scatter(name='x-shifts', x=np.arange(len(mc_shifts[:,1])), y=mc_shifts[:,1],
    line=dict(width=0.4, color='blue'), showlegend=True, row=1, col=1)
    fig.add_scatter(name='y-shifts', x=np.arange(len(mc_shifts[:,0])), y=mc_shifts[:,0],
    line=dict(width=0.4, color='blue'),  showlegend=True, row=2, col=1)
    return fig

def normalize_trace(trace):
    norm_trace =  (trace - trace.min()) / (trace.max() - trace.min())
    return norm_trace

def trace_comparison_figure(df, cell, detrend, behavior_flag):
    fig = get_layout_traces_comparison_fig(behavior_flag)
    fig = add_data_traces_comparison_fig(fig, df, cell, detrend, behavior_flag)
    return fig 

def get_layout_traces_comparison_fig(behavior_flag):
    rows_num = 1
    row_heights = [30]
    if behavior_flag:
        rows_num = 2
        row_heights += [3]
    fig = make_subplots \
        (
        rows = rows_num, cols = 1,
        vertical_spacing = 0.01, row_heights = row_heights, shared_xaxes=True, 
        y_title = 'A.U', x_title = "Frames [#]"
        )
    if behavior_flag:
        fig['layout']['yaxis'+str(2)]['title'] = 'Position'+'<br>'+'[cm]'
        fig['layout']['yaxis'+str(2)]['title'].update(font=dict(size=10))
        fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(height=600)
    fig.update_layout(showlegend=True)
    return fig 

def add_data_traces_comparison_fig(fig, df, cell, detrend, behavior_flag):
    df = df.iloc[10:]
    df = df.reset_index(drop=True)
    steps = [consts.RAW_TRACES_EXTRACTION, consts.MOTION_CORRECTION, consts.SPATIAL_FOOTPRINT]
    for i, step_name in enumerate(steps):
        step_col_prefix, trace_color, spike_color = pipeline_steps_featurs(step_name)
        trace = df[step_col_prefix + str(cell)]
        if detrend:
            trace = detrend_func(np.arange(len(trace)), trace)
        trace = normalize_trace(trace)
        trace = trace + i
        spike_times = df[df[consts.SPIKES + str(cell)] == 1].index
        spike_value = 1.01 * trace.iloc[spike_times]
        fig.add_scatter(name=" spikes", x=spike_times, y=spike_value, mode='markers', legendgroup =2, showlegend=i==0,
                        marker=dict(size=2.5, color=spike_color), visible='legendonly', row=1, col=1)
        fig.add_scatter(name=step_name, x=np.arange(len(trace)), y=trace,
                        line=dict(color=trace_color, width=0.4), showlegend=True, row=1, col=1)
    if behavior_flag:
        behave_col = df[consts.POSITION]
        lick_times = df[df[consts.VIR_LICK] == 1].index
        lick_value = 1.01 * behave_col[lick_times]
        fig.add_trace(go.Scatter(x=np.arange(len(behave_col)), y=behave_col, yaxis="y2",
                    showlegend=False, marker=dict(color='black')), row=2, col=1, )
        fig.add_scatter(name = " licks", x=lick_times, y=lick_value, mode='markers', marker=dict(
            size=2.5, color=spike_color), visible='legendonly', row=2, col=1)
    return fig

def display_FOV_images(mean_image, slm_patterns, volpy_data, cell_num=None):
    fig = plt.figure(figsize=(10, 20))
    # [left, bottom, width, height]
    ax1 = plt.axes([0.05, 0.66, 0.25, 0.22]) # mean_image
    ax2 = plt.axes([0.32, 0.66, 0.25, 0.22]) # slm patterns
    ax3 = plt.axes([ 0.59, 0.66, 0.25, 0.22]) # spatial_components
    pic_titels = ["Mean image", "SLM patterns", "Spatial Footprint"]
    for i, ax in enumerate([ax1,ax2,ax3]):
        ax.set_title(pic_titels[i], fontsize=14)
        ax.set_xticklabels([])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_yticklabels([])
    ax1.imshow(mean_image, cmap='gray')
    ax2.imshow(slm_patterns.sum(0), cmap='gray')
    for j, i in enumerate(slm_patterns): # annotate with cell numbers
        x = np.argwhere(i)
        x_c, y_c = x.mean(0)
        txt = ax2.text(y_c, x_c, str(j+1), ha='center', va='center', wrap=True)
        txt._get_wrap_line_width = lambda : 10
    if cell_num is not None:
        weights = volpy_data['weights'][cell_num-1]
        ax3.imshow(weights, interpolation='None', cmap=plt.cm.gray, vmax=np.max(weights)/2, vmin=0)
    else:
        weights = np.zeros_like(volpy_data['weights'][0])
        for i in range(len(slm_patterns)):
            footprint = volpy_data['weights'][i]
            weights += footprint
            ax3.imshow(weights, interpolation='None', cmap=plt.cm.gray, vmax=np.max(footprint)/2, vmin=0)
        weights[weights == 0] = np.nan
    return fig
