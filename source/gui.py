import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import sys
import subprocess
import datetime
import json
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths
from utils import pipeline_utils as pipe_utils
from utils import data_utils as data_utils
from utils import DB_utils as db_utils


NUMBER_INPUT = "number_input"
BOOLEAN_INPUT = "boolean_input"
LIST_INPUT = "list_input"
MC_PARAMS_TITLE = "**Motion Correction Parameters**"
PARTIAL_PARAMS_TITLE = "**Partial Video Parameters**"
SF_PARAMS_TITLE = "**Spatial Footprint Parameters**"
########## initialization ###########
class GUI_parameter:
    def __init__(self, name, param_type, default, list_vals=None, display_name=None):
        self.key = name
        if display_name is None:
            self.name = name
        else:
            self.name = display_name
        self.type = param_type
        self.default = default
        self.list_vals = list_vals
        self.help = self.get_help()
        self.st_widget = self.create_widget()
    
    def get_help(self):
        if self.name == consts.GSIG_FILT:
            return "Size of the kernel std for high pass spatial filtering (the kernel will be in the shape of (3 times gsig, 3 times gsig)"
        elif self.name == consts.MAX_SHIFTS_X:
            return "Maximum allowed rigid shift"
        elif self.name == consts.PW_RIGID:
            return "NoRMCorre stand for Non-Rigid Motion Correction. \
                    \nnon-piecewise rigid is faster and sometimes it will be sufficient"
        elif self.name == consts.MAX_DEVIATION_RIGID:
            return "Maximum deviation allowed for patch with respect to rigid shifts"
        elif self.name == consts.OVERLAPS_X:
            return "Overlap between patches (size of patch strides + overlaps.)"
        elif self.name == consts.STRIDES_X:
            return "Start a new patch for pw-rigid motion correction every n pixels"
        else:
            return None
    def create_widget(self):
        if self.type == "list_input":
            st.selectbox(self.name, index=self.default, key=self.key, options=self.list_vals, help=self.help)
        if self.type == "number_input":
            st.number_input(self.name, key=self.key, value=self.default, help=self.help)
        if self.type == "boolean_input":
            st.checkbox(self.name, key=self.key, value=self.default, help=self.help)

@st.cache_data
def init_pipeline_session():
    """
    Every new streamlit session, a new log dir will be created.
    than. a mangaer process will initilize and read the json objects that will be saved in this
    directory for running the pipelines.
    """
    session_time = datetime.datetime.now().strftime("%d-%m-%Y___%H-%M-%S")
    pipe_utils.mkdir(os.path.join(paths.PIPELINE_LOGS_DIR, session_time))
    pipeline_runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), paths.PIPELINE_RUNNER_SCRIPT)
    subprocess.Popen(["python", pipeline_runner_script, session_time]) 
    return session_time

def init_session_state():
    title_col1, title_col2, title_col3 = st.columns(3)
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    with title_col2:
        st.title('**Adam Lab Voltage Imaging** _Pipeline_ :sunglasses:')
    for key in ["raw_video_path", "cage_input", "mouse_name_input", "seq_input"]:
        if key not in st.session_state:
            st.session_state[key] = ""
    tab_names = ["**_Run_**", "**_Monitor_**", "**_Results_**", "**_Upload_**", "**_DB_**"]
    whitespace = 28   
    run_tab, monitor_tab, results_tab, upload_tab, db_tab = st.tabs([s.center(whitespace,"\u2001") for s in tab_names])
    return run_tab, monitor_tab, results_tab, upload_tab, db_tab

########## 1st Tab: run pipelines ###########
def display_mc_params():
    with st.expander(MC_PARAMS_TITLE):
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            gSig_filt = GUI_parameter(consts.GSIG_FILT, NUMBER_INPUT, 9)
            max_shifts_x = GUI_parameter(consts.MAX_SHIFTS_X, NUMBER_INPUT, 60)
            max_shifts_y = GUI_parameter(consts.MAX_SHIFTS_Y, NUMBER_INPUT, 60)
        with param_col2:
            place_holder = st.write("")
            place_holder2 = st.write("")
            pw_rigid = GUI_parameter(consts.PW_RIGID, BOOLEAN_INPUT, False)
            place_holder3 = st.write("")
            max_deviation_rigid = GUI_parameter(consts.MAX_DEVIATION_RIGID, NUMBER_INPUT, 3)
        with param_col3:
            overlaps_x = GUI_parameter(consts.OVERLAPS_X, NUMBER_INPUT, 32)
            overlaps_y = GUI_parameter(consts.OVERLAPS_Y, NUMBER_INPUT, 32)
            strides_x = GUI_parameter(consts.STRIDES_X, NUMBER_INPUT, 96)
            strides_y = GUI_parameter(consts.STRIDES_Y, NUMBER_INPUT, 96)
    return 

def display_spatial_footprint_params():
    with st.expander(SF_PARAMS_TITLE):
        sf_param_col1, sf_param_col2, sf_param_col3 = st.columns(3)
        with sf_param_col1:
            volpy_input = GUI_parameter(consts.VOLPY_INPUT, LIST_INPUT, 0, [consts.MC_INPUT, consts.DENOISED_INPUT])
            context_size = GUI_parameter(consts.CONTEXT_SIZE, NUMBER_INPUT, 20)
            censor_size = GUI_parameter(consts.CENSOR_SIZE, NUMBER_INPUT, 5)
            nPC_bg = GUI_parameter(consts.NPC_BG, NUMBER_INPUT, 8)
            ridge_bg = GUI_parameter(consts.RIDGE_BG, NUMBER_INPUT, 0.01)
        with sf_param_col2:
            hp_freq_pb = GUI_parameter(consts.HP_FREQ_PB, NUMBER_INPUT, 0.333)
            clip = GUI_parameter(consts.CLIP, NUMBER_INPUT, 200)
            min_spikes = GUI_parameter(consts.MIN_SPIKES, NUMBER_INPUT, 10)
            pnorm = GUI_parameter(consts.PNORM, NUMBER_INPUT, 0.5)
            threshold = GUI_parameter(consts.THRESHOLD, NUMBER_INPUT, 3)
        with sf_param_col3:
            sub_freq = GUI_parameter(consts.SUB_FREQ, NUMBER_INPUT, 50)
            n_iter = GUI_parameter(consts.N_ITER, NUMBER_INPUT, 2)
            threshold_method = GUI_parameter(consts.THRESHOLD_METHOD, LIST_INPUT, 0, [consts.ADAPTIVE_THRESHOLD, consts.SIMPLE_THRESHOLD])
            weight_update = GUI_parameter(consts.WEIGHT_UPDATE, LIST_INPUT, 0, [consts.RIDGE, consts.NMF])
            sf_place_holder = st.write("")
            sf_place_holder2 = st.write("")
            flip_signal = GUI_parameter(consts.FLIP_SIGNAL, BOOLEAN_INPUT, False)
            sf_place_holder2 = st.write("")
    return 

def display_partial_video_params():
    with st.expander(PARTIAL_PARAMS_TITLE):
        partial_col1, partial_col2, partial_col3 = st.columns(3)
        with partial_col1:
            place_holder = st.write("")
            place_holder2 = st.write("")
            partial_video = GUI_parameter(consts.PARTIAL_VIDEO, BOOLEAN_INPUT, False)
        with partial_col2:
            start_frame = GUI_parameter(consts.START_FRAME, NUMBER_INPUT, 1)
        with partial_col3:
            start_frame = GUI_parameter(consts.END_FRAME, NUMBER_INPUT, 1000)
    return 

def display_video_input():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1) # Make dialogbox appear on top of other windows
    video_input_col_1, video_input_col_2 = st.columns([7,1])
    with video_input_col_1:
        raw_video_path_text = st.text_input('**_Raw video path_**', st.session_state.raw_video_path)
    with video_input_col_2:
        place_holder = st.write("")
        place_holder2 = st.write("")
        browse_video = st.button('**_Browse_**')
    if browse_video:
        st.session_state.raw_video_path = filedialog.askopenfilename(master=root)
        cage, mouse_name, seq = pipe_utils.get_video_details(st.session_state.raw_video_path)
        st.session_state["cage_input"] = cage
        st.session_state["mouse_name_input"] = mouse_name
        st.session_state["seq_input"] = seq
        st.experimental_rerun()
    return 

def display_mouse_details():
    mouse_details_col_1, mouse_details_col_2, mouse_details_col_3, mouse_details_col_4 = st.columns(4)
    with mouse_details_col_1:
        cage = st.text_input('**_Cage_**', value=st.session_state.cage_input ,key=consts.CAGE)
    with mouse_details_col_2:
        mouse_name = st.text_input('**_Mouse name_**',value=st.session_state.mouse_name_input, key=consts.MOUSE_NAME)
    with mouse_details_col_3:
        imaging_seq = st.text_input('**_Imaging sequence_**',value=st.session_state.seq_input, key=consts.SEQ)
    with mouse_details_col_4:
        cell_type = st.selectbox('**_cell type_**', ['Pyr', 'IN', 'Pyr & IN'], index = 0, key=consts.CELL_TYPE)
    return 

def display_pipeline_steps():
    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns([1.2,2.5,2,2])
    with steps_col1:
        st.markdown('**_Pipeline steps:_**')
    with steps_col2:
        raw_traces_extraction = GUI_parameter(consts.RAW_TRACES_EXTRACTION, BOOLEAN_INPUT, True, display_name="**Raw Traces Extraction**")
        mc = GUI_parameter(consts.MOTION_CORRECTION, BOOLEAN_INPUT, True, display_name="**Motion Correction**")
    with steps_col3:
        spatial_footprint = GUI_parameter(consts.SPATIAL_FOOTPRINT, BOOLEAN_INPUT, True, display_name="**Spatial Footprint**")
        data_merger = GUI_parameter(consts.BEHAVIOR_AND_TRACES_MERGE, BOOLEAN_INPUT, True, display_name="**Data Merger**")
    with steps_col4:
        spike_detection = GUI_parameter(consts.SPIKE_DETECTION, BOOLEAN_INPUT, True, display_name="**Spike Detection**")
    return 

def run_pipeline_logic(session_time):
    run_col1, run_col2, run_col3, run_col4, run_col5 = st.columns(5)
    with run_col3:
        run_pipeline = st.button('**_run pipeline_**', type="primary")
    if run_pipeline:
        if not st.session_state.raw_video_path.endswith(".raw"):
            st.warning(' Enter a valid raw video path', icon="⚠️")
        else:
            save_pipeline_params(session_time)
    
def save_pipeline_params(session_time):
    gui_params = create_gui_params(dict(st.session_state))
    pipe_dir_name = "_".join([gui_params[consts.CAGE], gui_params[consts.MOUSE_NAME], gui_params[consts.SEQ], gui_params[consts.GUI_TIME]]) 
    pipe_dir = os.path.join(paths.PIPELINE_LOGS_DIR, session_time, pipe_dir_name)
    pipe_utils.mkdir(pipe_dir)
    param_file_path = os.path.join(pipe_dir, consts.PARAMS_FILE_SUFFIX_NAME[1:])
    with open(param_file_path , 'w') as fp:
        json.dump(gui_params, fp, indent=4)
    return 

def create_gui_params(gui_params):
    gui_time = datetime.datetime.now().strftime("%d-%m-%Y___%H-%M-%S")
    gui_params[consts.GUI_TIME] = gui_time
    gui_params[consts.RAW_VIDEO_PATH_LINUX] = pipe_utils.windows_to_linux_path(gui_params[consts.RAW_VIDEO_PATH])
    gui_params[consts.HOME_DIR_LINUX] = os.path.split(gui_params[consts.RAW_VIDEO_PATH_LINUX])[0]
    gui_params[consts.HOME_DIR] = os.path.split(gui_params[consts.RAW_VIDEO_PATH])[0]
    return gui_params     

########## 2nd Tab: pipelines progress ###########
def pipelines_progress(session_time):
    cols = st.columns([1,2,1])
    with cols[1]:
        st.subheader(':blue[**_Pipelines progress_**]')
    session_dir = os.path.join(paths.PIPELINE_LOGS_DIR, session_time)
    for i, pipe_dir_name in enumerate(os.listdir(session_dir)):
        pipe_dir = os.path.join(session_dir, pipe_dir_name)
        display_pipeline_progress(i, pipe_dir)
    
def display_pipeline_progress(pipe_id, pipe_dir):
    param_file = os.path.join(pipe_dir, consts.PARAMS_FILE_SUFFIX_NAME[1:])
    with open(param_file) as json_file:
        gui_params = json.load(json_file)
    pipe_name = get_pipe_name(pipe_id, gui_params)
    steps_num = get_total_steps_num(gui_params)
    comleted_steps = get_completed_steps(pipe_dir)
    st.progress(comleted_steps/steps_num , text=pipe_name)
    completed_steps, failed = display_pipeline_logs(pipe_dir)
    if failed:
        st.error("Pipeline failed", icon ="🚨")
    if completed_steps == steps_num:
        st.success("Pipeline completed!", icon="✅")

def get_pipe_name(pipe_id, gui_params):
    pipe_name = "**" + '# '+ str(pipe_id + 1) + ' ' + " ".join([gui_params[consts.CAGE], gui_params[consts.MOUSE_NAME], gui_params[consts.SEQ]]) + "**" 
    return pipe_name

def get_total_steps_num(gui_params):
    steps_num = 0
    steps_lst = pipe_utils.get_steps_lst()
    for step in steps_lst:
        if gui_params[step]:
            steps_num += 1
    return steps_num

def get_completed_steps(pipe_dir):
    completed_steps = 0 
    for file_name in os.listdir(pipe_dir):
        if file_name.endswith('.txt'):
            completed_steps += 1
    return completed_steps

def display_pipeline_logs(pipe_dir):
    completed_steps = 0
    failed = False
    log_files = filter(lambda file_name: os.path.join(pipe_dir, file_name).endswith('.txt'), os.listdir(pipe_dir))
    log_files_sorted = sorted(log_files, key = lambda file_name: os.path.getmtime(os.path.join(pipe_dir, file_name)))
    for log_file in log_files_sorted:
        step_name = os.path.splitext(log_file)[0]
        with st.expander("**_" + step_name + "_**"):
            with open(os.path.join(pipe_dir, log_file), 'r') as f:
                logs = [line.rstrip('\n') for line in f]
                log_state = logs[0]
                logs = " " + "\n".join(logs)
                if (consts.JOB_FAILED in log_state) or consts.JOB_CANCELLD in log_state:
                    st.error(logs, icon ="🚨")
                    failed = True
                elif consts.JOB_FINISHED in log_state:
                    st.success(logs, icon="✅")
                    completed_steps += 1
                else:    
                    st.info(logs, icon="ℹ️")
    return completed_steps, failed
    
########## 3rd Tab: pipelines results ###########
def display_mouse_details_for_results():
    mouse_details_col_1, mouse_details_col_2, mouse_details_col_3 = st.columns(3)
    with mouse_details_col_1:
        cage = st.text_input('**_Cage_**', value = "COP7")
    with mouse_details_col_2:
        mouse_name = st.text_input('**_Mouse name_**', value = "R1")
    with mouse_details_col_3:
        imaging_seq = st.text_input('**_Imaging sequence_**', value = "7_h")
    return cage, mouse_name, imaging_seq

@st.cache_data
def get_fov_objects(cage, mouse_name, seq):
    df = data_utils.get_fov_data(cage, mouse_name, seq)
    cells_list = data_utils.get_cells_list(df)
    behavior_flag = data_utils.check_behavioral_data(df)
    volpy_data, slm_patterns, mean_image, mc_shifts = pipe_utils.get_pipline_results_data(cage, mouse_name, seq)
    return df, cells_list, behavior_flag, volpy_data, slm_patterns, mean_image, mc_shifts

def results_menu():
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        display_options = st.radio("Display options:", ["Cell by Cell", "Full FOV"])
    with res_col3:
        place_holder = st.write("")
        place_holder2 = st.write("")
        detrend_flag = st.checkbox('Detrend', True)
    return display_options, detrend_flag, res_col2

def display_results(display_options, detrend_flag, res_col2, cage, mouse_name, seq):
    if display_options == "Full FOV":
        all_cells_results(detrend_flag, res_col2, cage, mouse_name, seq)
    if display_options == "Cell by Cell":
        cell_by_cell_comparison(detrend_flag, res_col2, cage, mouse_name, seq)
    return

def all_cells_results(detrend, res_col2, cage, mouse_name, seq):
    with res_col2:
        display_step = st.selectbox("Pipeline step:", [None, consts.RAW_TRACES_EXTRACTION, consts.MOTION_CORRECTION, consts.SPATIAL_FOOTPRINT, consts.MC_SHIFTS])
    if display_step is not None:
        df, cells_list, behavior_flag, volpy_data, slm_patterns, mean_image, mc_shifts = get_fov_objects(cage, mouse_name, seq)
        images_fig = data_utils.display_FOV_images(mean_image, slm_patterns, volpy_data)
        st.pyplot(images_fig)
        st.write(cage, mouse_name, seq)
        if display_step == consts.MC_SHIFTS:
            fig = data_utils.mc_shifts_fig(mc_shifts)
        else:
            fig = data_utils.fov_traces_figure(df, cells_list, behavior_flag, detrend, display_step)
        st.plotly_chart(fig, use_container_width=True)
    return 

def cell_by_cell_comparison(detrend, res_col2, cage, mouse_name, seq):
    if cage != "" and mouse_name != "" and seq != "":
        df, cells_list, behavior_flag, volpy_data, slm_patterns, mean_image, mc_shifts = get_fov_objects(cage, mouse_name, seq)
        placeholder = st.empty()
        with placeholder:
            images_fig = data_utils.display_FOV_images(mean_image, slm_patterns, volpy_data)
            st.pyplot(images_fig)
        with res_col2:
            cell_num = st.selectbox("Cell number:", [None] + [i for i in cells_list])
        if cell_num is not None:
            with placeholder:
                images_fig = data_utils.display_FOV_images(mean_image, slm_patterns, volpy_data, cell_num)
                st.pyplot(images_fig)
            st.write(cage, mouse_name, seq)
            fig = data_utils.trace_comparison_figure(df, cell_num, detrend, behavior_flag)
            st.plotly_chart(fig, use_container_width=True)
    return 

########## 4th Tab: Upload to DB ###########
def display_table(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    gb.configure_grid_options(enableRangeSelection=True)
    gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[], groupSelectsChildren="Group checkbox select children")
    table = AgGrid(df, gridOptions=gb.build(), data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=True, enable_enterprise_modules=True, allow_unsafe_jscode=True, columns_auto_size_mode=True)
    return table

def remove_from_queue(table):
    if len(table['selected_rows']) == 0:
        return
    data = table['data'].to_dict(orient='records')
    row = table['selected_rows'][0]
    row.pop('_selectedRowNodeInfo', None)
    index = data.index(row)
    queue_df = pd.read_csv(paths.DB_UPLOAD_QUEUE_PATH)
    queue_df = queue_df.drop(index, axis=0)
    queue_df.to_csv(paths.DB_UPLOAD_QUEUE_PATH, index=False)

def get_experiment_details(experiment):
    cage = experiment[0][db_utils.CAGE]
    mouse_name = experiment[0][db_utils.MOUSE_NAME]
    seq = str(experiment[0][db_utils.SEQ])
    video_path = experiment[0][db_utils.VIDEO_PATH]
    experiment_date = experiment[0][db_utils.EXPERIMENT_DATE]
    cell_type = experiment[0][db_utils.CELL_TYPE]
    return cage, mouse_name, seq, video_path, experiment_date, cell_type

def laps_removal_manager(df, laps_to_remove, cage, mouse_name, seq, video_path, experiment_date, cell_type):
    df = data_utils.remove_lpas(df, laps_to_remove)
    # saving new file
    new_suffix = seq + consts.REMOVED_LAPS
    new_file_name = os.path.join(paths.DATASET_DIR_WINDOWS, cage, mouse_name, new_suffix + '.parquet')
    df.to_parquet(new_file_name, index=False)
    # update upload db queue
    pipe_utils.save_record_to_DB_queue(experiment_date, cage, mouse_name, new_suffix, cell_type, video_path)
    st.experimental_rerun()
    return

def init_queue_table():
    cols = st.columns(5)
    with cols[2]:
        st.subheader(':blue[**_Upload Queue_**]')
    upload_queue_df = pd.read_csv(paths.DB_UPLOAD_QUEUE_PATH)
    upload_table = display_table(upload_queue_df)
    return upload_table

def queue_removal_button(upload_table):
        if st.button('Remove from queue'):
            remove_from_queue(upload_table)
            st.experimental_rerun()

def get_experiment_data(queue_row):
    cage, mouse_name, seq, video_path, experiment_date, cell_type = get_experiment_details(queue_row)
    df = data_utils.get_parquet_data(cage, mouse_name, seq)
    cells_list = data_utils.get_cells_list(df)
    frame_rate = pipe_utils.get_frame_rate(video_path)
    fov = pipe_utils.get_fov_name(video_path)
    remapping = pipe_utils.is_reemapping_experiment(df)
    removed_laps = False
    if consts.REMOVED_LAPS in seq:
        removed_laps = True
    return cage, mouse_name, seq, video_path, experiment_date, cell_type, \
            df, cells_list, frame_rate, fov, remapping, removed_laps

def display_experiment_cells(df, cells_list):
    fig = data_utils.fov_traces_figure(df, cells_list, True, True, consts.SPATIAL_FOOTPRINT)
    st.plotly_chart(fig, use_container_width=True)


def lap_removal(df, cage, mouse_name, seq, video_path, experiment_date, cell_type):
    with st.expander("**laps removal**"):
        with st.form("laps_removal_form"):
            laps_to_remove = st.multiselect('Laps to remove:',(list(np.arange(1, df[consts.LAP_COUNTER].nunique() + 1))))
            if st.form_submit_button("remove laps"):
                if len(laps_to_remove) > 0:
                    laps_removal_manager(df, laps_to_remove, cage, mouse_name, seq, video_path, experiment_date, cell_type)

def upload_data_display(cells_lst, queue_table, experiment_date, cage, mouse_name, seq, cell_type, fov, frame_rate, remapping, removed_laps, video_path):
    with st.expander("**Good cells selection & comments**"):
        with st.form("db_upload_form"):
            good_cells = st.text_input('Good cells', value = str(cells_lst))
            comments = st.text_input('Comments', value = '')
            upload_cols = st.columns(5)
            with upload_cols[2]:
                if st.form_submit_button('Upload!', type="primary"):
                    new_record = [experiment_date, cage, mouse_name, seq, good_cells, cell_type, fov, frame_rate, remapping, removed_laps, comments, video_path]
                    upload_experiment_to_db(new_record, queue_table)

def upload_experiment_to_db(new_record, queue_table):
    pipe_utils.save_record_to_DB(new_record)
    remove_from_queue(queue_table)
    st.experimental_rerun()

########## 5th Tab: DB ###########
def diplay_db_table():
    db_manager = db_utils.DB_manager()
    table = display_table(db_manager.db)
    return table, db_manager

def update_db(table):
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Save changes to DB'):
            save_db_data_after_user_change(table['data'])
            st.experimental_rerun()

def save_db_data_after_user_change(data):
    """
    Get as an input the whole DB after some cell (or cells) beieng edited by user.
    save the new DB in the main path, and keep copy of the old DB for backup.
    """
    with open(os.path.join(paths.DB_BACKUPS_PATH, 'seq.txt'), 'r') as f:
        seq = f.readline()
    with open(os.path.join(paths.DB_BACKUPS_PATH, 'seq.txt'), 'w') as f:
        f.write(str(int(seq)+1))

    os.rename(paths.DB_PATH, os.path.join(paths.DB_BACKUPS_PATH, 'backup_' + seq + '.csv'))
    data = data.drop(db_utils.SESSIONS_COUNTER, axis=1)
    data.to_csv(paths.DB_PATH, index=False, header=False)

def analysis_options_manager(table, db_manager):
    cols = st.columns(3)
    with cols[0]:
        analysis_subject = st.radio("What do you want to analyze?", ['FOV', 'DB'])
    with cols[1]:
        if analysis_subject == "FOV":
            analysis_option = single_fov_analysis_options(db_manager)
        if analysis_subject == "DB":
            analysis_option = all_DB_analysis_option(db_manager)
    display_analysis_figure(analysis_subject, table, analysis_option, db_manager)

def  display_analysis_figure(analysis_subject, table, analysis_option, db_manager):
    if analysis_subject == "FOV":
        display_single_fov_analysis_figures(table, analysis_option, db_manager)
    if analysis_subject == "DB":
        display_all_DB_analysis_figures(table, analysis_option, db_manager)

def single_fov_analysis_options(db_manager):
    help_message = "Select a row from the DB, and choose an analysis method from the selectbox."
    description = "Choose an analysis method for specific FOV:"
    analysis_option = st.selectbox(description, [None] + db_manager.fov_analysis_options, help=help_message)
    return analysis_option

def all_DB_analysis_option(db_manager):
    help_message = "This section is an analysis that doesn't depends on specific FOV."
    description = "Choose an analysis method for all the DB:"
    analysis_option = st.selectbox(description, [None] + db_manager.entire_db_analysis_options, help=help_message)
    return analysis_option

def display_single_fov_analysis_figures(table, analysis_option, db_manager):
    selected_row = table['selected_rows']
    if analysis_option is not None:
        if len(selected_row) > 0:
            if selected_row[0][db_utils.GOOD_CELLS] != "None":
                fig = db_manager.display_single_FOV_figure(analysis_option, selected_row[0])       
                st.plotly_chart(fig, use_container_width=True)
                images_fig = db_manager.display_FOV_images(selected_row[0])
                st.pyplot(images_fig) 
            else:
                st.write("This FOV doesn't contains any good cells.")
        else:
            st.write("Choose a record from the DB to analyze")

def display_all_DB_analysis_figures(table, analysis_option, db_manager):
    if analysis_option == "Population firing rate":
        col1, col2, col3 = st.columns(3)
        with col1:
            population = st.radio("Choose cells' type:", ('Pyr', 'IN',))
        with col2:
            place_feild_flag = st.radio("Wich cells:", ('all', 'with place feild',))
        with col3:
            bins_number = st.slider('Choose the number of bins:', 40, 200, 60, 20, )
        fig = set_and_get_population_firing_rate_fig(population, bins_number, place_feild_flag)
        population_fr_plot = st.plotly_chart(fig, use_container_width=True)
        heatmap_fig = set_and_get_heatmap_firing_rate_fig( population, bins_number, place_feild_flag)
        heatmap_display = st.plotly_chart( heatmap_fig, use_container_width=True)

########### Tabs wrapers #############
def display_run_pipeline_tab(run_pipeline_tab, session_time):
    with run_pipeline_tab:
        cols = st.columns([1,2,1])
        with cols[1]:
            display_mc_params()
            display_spatial_footprint_params()
            display_partial_video_params()
            display_video_input()
            display_mouse_details()
            display_pipeline_steps()
            run_pipeline_logic(session_time)

def display_monitor_tab(monitor_tab, session_time):
    with monitor_tab:
        cols = st.columns([1,2,1])
        with cols[1]:
            pipelines_progress(session_time)

def display_results_submitted():
    st.session_state.submitted = True

def display_results_tab(results_tab):
    with results_tab:
        with st.form("results form"):
            cols = st.columns(3)
            with cols[1]:
                cage, mouse_name, seq = display_mouse_details_for_results()
                cols2 = st.columns(3)
                with cols2[1]:
                    st.form_submit_button("Display", on_click=display_results_submitted)
        if st.session_state.submitted:
            display_options, detrend_flag, res_col2 = results_menu()
            display_results(display_options, detrend_flag, res_col2, cage, mouse_name, seq)

def display_upload_tab(upload_tab):
    with upload_tab:
        upload_table = init_queue_table()
        queue_removal_button(upload_table)
        if len(upload_table['selected_rows']) > 0:
            cage, mouse_name, seq, video_path, experiment_date, cell_type, \
            df, cells_list, frame_rate, fov, remapping, removed_laps = \
            get_experiment_data(upload_table['selected_rows'])
            display_experiment_cells(df, cells_list)
            lap_removal(df, cage, mouse_name, seq, video_path, experiment_date, cell_type)
            upload_data_display(cells_list, upload_table, experiment_date, cage, mouse_name, seq,  
                                cell_type, fov, frame_rate, remapping, removed_laps, video_path)
              

def display_db_tab(db_tab):
    with db_tab:
        table, db_manager = diplay_db_table()
        update_db(table)
        analysis_options_manager(table, db_manager)

def main():
    import importlib
    importlib.reload(consts)
    importlib.reload(data_utils)
    st.set_page_config(layout="wide")
    session_time = init_pipeline_session()
    run_pipeline_tab, monitor_tab, results_tab, upload_tab, db_tab = init_session_state()
    display_run_pipeline_tab(run_pipeline_tab, session_time)
    display_monitor_tab(monitor_tab, session_time)
    display_results_tab(results_tab)
    display_upload_tab(upload_tab)
    display_db_tab(db_tab)

if __name__ == "__main__":
    main()

