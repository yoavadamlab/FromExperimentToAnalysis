import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import files_paths as paths
import pipeline_constants as consts
import data_utils as data_utils
import pipeline_utils as pipe_utils
import ast
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

### columns names ###
EXPERIMENT_DATE = "experiment_date"
CAGE = "cage"
MOUSE_NAME = "mouse_name"
SEQ = "pipeline_seq"
GOOD_CELLS = "good_cells"
CELL_TYPE = "cell_type"
FOV = "FOV"
FRAME_RATE = "frame_rate"
REMAPPING = "remapping"
REMOVED_LAPS = "removed_laps"
COMMENTS = "comments"
VIDEO_PATH = "video_path"
SESSIONS_COUNTER = "sessions_counter"
CELLS_NUM = "number_of_cells"
### analysis options ###
CELLS_ACTIVITY = "Cells' activity"
FR_AND_SUB = "Firing rate & subthreshold"
LAP_FR = "Firing per lap"
REMAPP_ANALYSIS = "Remapping analysis"
ACTIVITY_PER_LAP = "Activity per lap"
LONGITUDIAL_ANALYSIS = "Longitudinal analysis"
FR_POPULATION = "Population firing rate"
SPIKE_DETECTION = "Spike detection"
### analysis columns  #####
BIN = "binned_position"
TIME_IN_BIN_PER_LAP = 'time_in_bin_per_lap'
FT_PREFIX = "fr_cell_"
MEAN_FR_PREFIX = "mean_fr_over_laps_cell_"
SEM_FR_PREFIX = "sem_fr_over_laps_cell_"
NON_SPIKED_TRACE_PREFIX = "non_spiked_trace_cell_"
SUB_ACTIVITY_PREFIX = "sub_activity_cell_"
SEM_SUB_ACTIVITY_PREFIX = "sem_sub_activity_cell_"
                                    
class DB_manager:
    def __init__(self):
        self.schema = self._get_schema()
        self.db = self._get_DB()
        self.fov_analysis_options = self._get_single_fov_analysis_options()
        self.entire_db_analysis_options = self._get_entire_db_analysis_options()
        self.fig_generator = FigGenerator(self.schema, self.db)

    def _get_schema(self):
        return [
            EXPERIMENT_DATE, CAGE, MOUSE_NAME, SEQ,  GOOD_CELLS, CELL_TYPE, 
            FOV, FRAME_RATE, REMAPPING, REMOVED_LAPS, COMMENTS, VIDEO_PATH
        ]

    def _get_DB(self):
        df = pd.read_csv(paths.DB_PATH, names=self.schema, index_col=False)
        df = self._append_DB_columns(df)
        df = df.sort_values(EXPERIMENT_DATE, ascending=False)
        return df

    def _append_DB_columns(self, df):
        df = self._add_longitudinal_index(df)
        return df

    def _add_longitudinal_index(self, df):
        df[SESSIONS_COUNTER] = df.groupby([CAGE, MOUSE_NAME, FOV])[EXPERIMENT_DATE].transform('nunique')
        return df

    def _get_single_fov_analysis_options(self):
        """
        return list of analysis options relvenat to a single FOV
        """
        return [
            CELLS_ACTIVITY, FR_AND_SUB, LAP_FR,  
            ACTIVITY_PER_LAP, LONGITUDIAL_ANALYSIS, SPIKE_DETECTION
        ]

    def _get_entire_db_analysis_options(self):
        """
        return list of analysis options relvenat to the entire DB - 
        meaning some summary figure over all the dataset
        """
        return [FR_POPULATION]

    def display_single_FOV_figure(self, fig_name, data_record):
        experiment = Experiment(self.schema, data_record)
        fig = self.fig_generator.create_fig(fig_name, experiment)
        return fig
    
    def display_FOV_images(self, data_record, fig_name=None):
        experiment = Experiment(self.schema, data_record)
        fig = self.fig_generator.create_images_fig(fig_name, experiment)
        return fig

    def get_single_record(self, cage, mouse_name, pipeline_seq):
        single_record = self.db[(self.db[CAGE]==cage) & (self.db[MOUSE_NAME]==mouse_name) & (self.db[SEQ]==pipeline_seq)]
        return single_record

    def get_single_experiment(self,cage,mouse_name,pipeline_seq):
        single_record = self.get_single_record(cage,mouse_name,pipeline_seq)
        db_record = {}
        for i,j in zip(self.schema, single_record.values[0]):
            db_record[i] = j
        exp = Experiment(self.schema, db_record)
        return exp

class Experiment:
    def __init__(self, db_schema, db_record):
        self.metadata = self._extract_metadata_from_db_record(db_schema, db_record)
        self.raw_data = self._get_experiment_data()
        self.data = self.preprocessing(self.raw_data)
        self.behavior_flag = data_utils.check_behavioral_data(self.data)
        self.cells = self._create_cells()

    def _create_cells(self):
        cells = {}
        for cell_num in self.metadata[GOOD_CELLS]:
            cell = Cell(self, cell_num)
            cells[cell_num] = cell
        return cells
        
    def _extract_metadata_from_db_record(self, db_schema, db_record):
        metadata = {}
        for feild in db_schema:
            if feild == GOOD_CELLS:
                good_cells = [n for n in ast.literal_eval(db_record[feild])]
                good_cells.sort()
                metadata[feild] = good_cells
                metadata[CELLS_NUM] = len(good_cells)
                continue
            if feild == REMAPPING:
                val = db_record[feild]
                if isinstance(val, bool):
                    metadata[feild] = val
                else:
                    metadata[feild] = val.lower() == "true"
                continue
            metadata[feild] = db_record[feild]
        return metadata
    
    def _get_experiment_data(self):
        """
        get the merged_data object.
        slice out the first 1000 rows.
        """
        try:
            data_file = os.path.join(paths.DATASET_DIR_WINDOWS, self.metadata[CAGE], self.metadata[MOUSE_NAME], self.metadata[SEQ] + '.parquet')
            df = pd.read_parquet(data_file)
        except:
            data_file = os.path.join(paths.DATASET_DIR, self.metadata[CAGE], self.metadata[MOUSE_NAME], self.metadata[SEQ] + '.parquet')
            df = pd.read_parquet(data_file)
        return df
    
    def preprocessing(self, df):
        # 1. cut first 1000 frames detrend traces and transform to delta F
        df = self.preprocess_traces(df)
        # for behavioral experiments:
        if data_utils.check_behavioral_data(df):
            # 1. delete bad laps 
            df["min_position"] = df.groupby(consts.LAP_COUNTER)[consts.POSITION].transform(min)  # new column for removing bad laps (with position under -30)
            df = df[df["min_position"] > -35] # virmen unit that should't be crossed
            # 3. slice out all frames before the animal started to walk
            df["chnged_position"] = df[consts.POSITION].astype(float).diff()
            start = df[df["chnged_position"] != 0].index[1]
            df = df.iloc[start:]
            df.reset_index(inplace=True, drop=True)
            # 4. slice out first and last laps
            df = self._slice_out_first_and_last_laps(df)
        return df
    
    def transform_to_df_over_f(self, trace):
        trace = trace + (2*np.abs(trace.min()))  # prevent negative values
        F0 = trace[-1000:].mean()
        trace = (trace - F0) / F0
        return trace
    
    def preprocess_traces(self, df):
        df = df.iloc[1000:, :]
        df.reset_index(inplace=True, drop=True)
        trace_col_prefix = data_utils.get_trace_col_prefix(df)
        for cell in self.metadata[GOOD_CELLS]:
            trace = df[trace_col_prefix + str(cell)]
            try: # try to detrend if doesnt work keep the non detrend trace
                trace = data_utils.detrend_func(np.arange(len(trace)), trace)
            except:
                print("detrend problem with cell number: ", cell)
            trace = self.transform_to_df_over_f(trace)
            df[trace_col_prefix + str(cell)] = trace
        return df

    def get_traces(self, df=None):
        """
        return np.array in the shape: #cells, #frames
        """
        if df is None:
            df = self.data
        trace_col_prefix = data_utils.get_trace_col_prefix(df)
        traces = np.empty((self.metadata[CELLS_NUM], len(df)))
        for i, cell in enumerate(self.metadata[GOOD_CELLS]):
            trace = df[trace_col_prefix + str(cell)]
            traces[i] = trace
        return traces

    def _slice_out_first_and_last_laps(self, df):
        """
        The first lap can be trimmed from slicing out the first 1000 
        frames in the preprocessing step.
        The lap can halt in the middle due to imaging time.
        Since those laps are not complete they mess the data and
        rge statistics and better to be removed.
        """
        df = df[df[consts.LAP_COUNTER] > df[consts.LAP_COUNTER].min()]
        df = df[df[consts.LAP_COUNTER] < df[consts.LAP_COUNTER].max()]
        df[consts.LAP_COUNTER] = df[consts.LAP_COUNTER] - df[consts.LAP_COUNTER].min() + 1
        df = df.reset_index(drop=True)
        return df

    def get_fam_and_novel_df(self, df=None):
        if df is None:
            df = self.data
        novel_start = df[df[consts.WORLD] == 3].index[0]
        df_fam = df.iloc[:novel_start]
        df_fam = df_fam.reset_index(drop=True)
        df_nov = df.iloc[novel_start:]
        df_nov = df_nov.reset_index(drop=True)
        return df_fam, df_nov

    def bin_data_by_position(self, bins_num, df):
        df[BIN], _ = pd.cut(df[consts.POSITION], bins=bins_num, labels=np.arange(bins_num), include_lowest=True, retbins=True)
        df[BIN] = pd.to_numeric(df[BIN])
        return df  

    def calculate_bin_length(self, bins_num, df):
        mean_lap_length = int(df.groupby(consts.LAP_COUNTER)[consts.LAP_LEN_CUMSUM].max().drop_duplicates().mean() / 10)
        cm_per_bin = round(mean_lap_length / bins_num, 2)
        return cm_per_bin

    def get_firing_rate_per_lap(self, bins_num, df=None):
        if df is None:
            df = self.data
        # calculate time in bin per lap column
        df = self.bin_data_by_position(bins_num, df)
        laps_dfs = df.groupby([consts.LAP_COUNTER, BIN])
        df[TIME_IN_BIN_PER_LAP] = laps_dfs[consts.TS_TIME].transform(max) - laps_dfs[consts.TS_TIME].transform(min) + 0.0001
        # count spikes per bin per lap
        spike_col_prefix = data_utils.get_spike_col_prefix(df)
        spikes = [spike_col_prefix + str(i) for i in self.metadata[GOOD_CELLS]]
        spikes_in_bin_per_lap_cols = ["spikes_count_per_lap_per_bin_cell_" + str(i) for i in self.metadata[GOOD_CELLS]]
        df[spikes_in_bin_per_lap_cols] = laps_dfs[spikes].transform('sum')
        # keep one row only per lap per bin
        df = df[[consts.LAP_COUNTER, TIME_IN_BIN_PER_LAP, BIN] + spikes_in_bin_per_lap_cols].drop_duplicates()
        # calculate firing rate
        fr_cols = [FT_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        df[fr_cols] = df[spikes_in_bin_per_lap_cols].div(df[TIME_IN_BIN_PER_LAP], axis=0)
        df = df[[consts.LAP_COUNTER, BIN] + fr_cols].drop_duplicates()
        df = df.interpolate() # just filling the nan values with interpolation
        df.reset_index(drop=True, inplace=True)
        df.sort_values(BIN, inplace=True)
        return df
    
    def get_mean_firing_rate(self, bins_num, df=None):
        if df is None:
            df = self.data
        df = self.get_firing_rate_per_lap(bins_num, df)
        # cols names
        fr_cols = [FT_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        mean_fr_cols = [MEAN_FR_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        sem_fr_cols = [SEM_FR_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        # calc fr
        df[mean_fr_cols] = df.groupby([BIN])[fr_cols].transform('mean') * 1000  # for Hz units
        laps_number = self.get_laps_number(df)
        df[sem_fr_cols] = df.groupby([BIN])[fr_cols].transform('std') * 1000 * (1/np.sqrt(laps_number))  # for Hz unit
        df = df[[BIN] +  mean_fr_cols + sem_fr_cols].drop_duplicates()
        df = df.interpolate() # just filling the nan values with interpolation
        df.reset_index(drop=True, inplace=True)
        df.sort_values(BIN, inplace=True)
        return df
    
    def remove_spike_activty(self, df, window_size=5):
        traces = self.get_traces(df)
        spikes = self.get_spikes_timming(df)
        for i, cell in enumerate(self.metadata[GOOD_CELLS]):
            trace = traces[i]
            for spike_time in spikes[i]:
                start_index = max(0, spike_time - window_size)
                end_index = min(len(trace) - 1, spike_time + window_size)
                trace[start_index:end_index+1] = np.nan
            indices = np.arange(len(trace))
            trace = np.interp(indices, indices[~np.isnan(trace)], trace[~np.isnan(trace)])
            df[NON_SPIKED_TRACE_PREFIX + str(cell)] = trace 
        return df

    def get_subthreshold_activity(self, bins_num, df=None):
        if df is None:
            df = self.data
        df = self.bin_data_by_position(bins_num, df)
        df = self.remove_spike_activty(df)
        sub_activity_cols = [NON_SPIKED_TRACE_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        mean_sub_cols = [SUB_ACTIVITY_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        sem_sub_cols = [SEM_SUB_ACTIVITY_PREFIX + str(i) for i in self.metadata[GOOD_CELLS]]
        # calc fr
        df[mean_sub_cols] = df.groupby([BIN])[sub_activity_cols].transform('mean')
        laps_number = self.get_laps_number(df)
        df[sem_sub_cols] = df.groupby([BIN])[sub_activity_cols].transform('std') * (1/np.sqrt(laps_number))  
        df = df[[BIN] +  mean_sub_cols + sem_sub_cols].drop_duplicates()
        df = df.interpolate() # just filling the nan values with interpolation
        df.reset_index(drop=True, inplace=True)
        df.sort_values(BIN, inplace=True)
        return df

    def get_traces_without_spikes(self,window_size=5, df=None):
        if df is None:
            df = self.data
        removed_df = self.remove_spike_activty(df, window_size)
        subthresh_col_prefix = NON_SPIKED_TRACE_PREFIX
        traces = np.empty((self.metadata[CELLS_NUM], len(removed_df)))
        for i, cell in enumerate(self.metadata[GOOD_CELLS]):
            trace = removed_df[subthresh_col_prefix + str(cell)]
            traces[i] = trace
        return traces
        
    def get_spikes_timming(self, df=None):
        if df is None:
            df = self.data
        spike_col_prefix = data_utils.get_spike_col_prefix(df)
        spikes_timmimg = []
        for cell in self.metadata[GOOD_CELLS]:
            spikes_time = df[df[spike_col_prefix + str(cell)] == 1].index
            spikes_timmimg.append(spikes_time)
        return spikes_timmimg

    def get_spikes_height(self, spikes_timming, traces):
        heights = []
        for i, spikes_time in enumerate(spikes_timming):
            spikes_height = 1.01 * traces[i][spikes_time]
            heights.append(spikes_height)
        return heights
    
    def get_position(self, df=None):
        if df is None:
            df = self.data
        return df[consts.POSITION]
    
    def get_licks_timming(self, df=None):
        if df is None:
            df = self.data
        return df[df[consts.VIR_LICK] == 1].index
    
    def get_laps_number(self, df):
        laps_number = df[consts.LAP_COUNTER].nunique()
        return laps_number
    
    def get_reward_zone_bins(self, df, bins_num):
        df = self.bin_data_by_position(bins_num, df)
        rwd_df = df[[consts.POSITION, BIN]]
        start_rwd = 107 # hard coded fron VIRMEN file
        end_rwd = 128 
        start_rwd_bin = rwd_df[(rwd_df[consts.POSITION] >= start_rwd) & (rwd_df[consts.POSITION] < end_rwd)][BIN].min()
        end_rwd_bin = rwd_df[(rwd_df[consts.POSITION] >= start_rwd) & ( rwd_df[consts.POSITION] <= end_rwd)][BIN].max()
        return start_rwd_bin, end_rwd_bin
    
    def get_cell_idx(self, cell_num):
        for i, cell_number in enumerate(self.metadata[GOOD_CELLS]):
            if cell_number == cell_num:
                return i
            
    def fr_per_lap_df_to_fr_matrices(self, fr_per_lap_df, bins_num=40):
        fr_per_lap_df.sort_index(inplace=True)
        # create an empty dictionary to store the numpy arrays
        FR_mats = {}
        # iterate over the cell columns in the DataFrame
        for col_name in fr_per_lap_df.columns:
            if col_name.startswith('fr_cell_'):
                # pivot the DataFrame on lap_counter and binned_position for the current cell column
                pivot_df = fr_per_lap_df.pivot_table(index='lap_counter', columns='binned_position', values=col_name)
                # convert the pivot table to a numpy array
                FR_matrix = pivot_df.to_numpy()
                FR_matrix = np.nan_to_num(FR_matrix)
                # store the numpy array in the dictionary using the cell name as the key
                cell_name = col_name.replace('fr_cell_', '')
                FR_mats[cell_name] = FR_matrix
        return FR_mats
    
    def get_fr_matrices(self, bins_num=40):
        fr_per_lap_df = self.get_firing_rate_per_lap(bins_num)
        return self.fr_per_lap_df_to_fr_matrices(fr_per_lap_df, bins_num)

class Cell:
    def __init__(self, exp, cell_num):
        self.metadata = exp.metadata
        self.trace = self._get_trace(exp, cell_num)
        self.spikes = self._get_spikes(exp, cell_num)
        # self.FR_matrix = self._get_FR_matrix(exp, cell_num)
        self.name = f'{self.metadata[CAGE]}_{self.metadata[MOUSE_NAME]}_{self.metadata[SEQ]}_{cell_num}'

    def _get_trace(self, exp, cell_num):
        traces = exp.get_traces()
        cell_idx = exp.get_cell_idx(cell_num)
        return traces[cell_idx]
    
    def _get_spikes(self, exp, cell_num):
        spikes_timming = exp.get_spikes_timming()
        cell_idx = exp.get_cell_idx(cell_num)
        return spikes_timming[cell_idx]
    
    def _get_FR_matrix(self, exp, cell_num):
        # inefficent. a solution can be loading from existing path 
        # or by changing get_firing_rate_per_lap function to get a cell number as an input
        FR_matrices = exp.get_fr_matrices() 
        cell_idx = exp.get_cell_idx(cell_num)
        return FR_matrices[cell_idx]
    
    def get_some_statistics(self, path):
        #load the data from path by cell.nam
        pass


class FigGenerator:
    def __init__(self, schema, db):
        self.schema = schema
        self.db = db
        self.CELL_NUMBER_PREFIX = "cell # "
        self.AX_TITLE_FRAMES = "Frames [#]"
        self.AX_TITLE_AU = "A.U"
        self.AX_TITLE_POSITION = 'Position'+'<br>'+'[cm]'
        self.AX_TITLE_FR = "Firing rate <br> [Hz]"
        self.CELL = "cell "
        self.SPIKES = " spikes"
        self.PYR = "Pyr"
        self.IN = "IN"
        self.CM_PER_BIN = " cm per bin"
        self.DELTA_F = "&#916;F/F"
        self.LOW_SEM = "low_sem"

    def get_longitundinal_metadata(self, experiment):
        long_metadata = self.db.loc[
            (self.db[CAGE] == experiment.metadata[CAGE]) & 
            (self.db[MOUSE_NAME] == experiment.metadata[MOUSE_NAME]) &
            (self.db[FOV] == experiment.metadata[FOV])
            ]
        return long_metadata
            
    def get_longitudinal_data(self, experiment, cell_num):
        long_metadata = self.get_longitundinal_metadata(experiment)
        long_experiments = []
        for i, record in long_metadata.iterrows():
            if str(cell_num) in record[GOOD_CELLS]:
                db_record = {}
                for field, value in zip(self.schema, record):
                    if field == GOOD_CELLS:
                        value = str(value)
                    db_record[field] = value
                experiment = Experiment(self.schema, db_record)
                long_experiments.append(experiment)
        return long_experiments
    
    def get_generic_title(self, experiment):
        fig_title1 = experiment.metadata[CAGE] + " " + experiment.metadata[MOUSE_NAME] + " " + experiment.metadata[FOV]
        fig_title2 = "Pipeline seq: " + experiment.metadata[SEQ]
        return fig_title1 + "<br>" + fig_title2

    def get_axis_limit(self, trace, top_space, low_space):
        trace_ax_max = trace.max() + top_space * (trace.max() - trace.min())
        trace_ax_min = trace.min() - low_space * (trace.max() - trace.min())
        return trace_ax_min, trace_ax_max
    
    def get_range_edges_by_line_name(self, fig, line_name):
        """
        for generate button with two modes - relative scale
        and common scale to all subplots, dicts with edges 
        per each axis need to ne extracted
        """
        fig_min = float("inf")
        fig_max = float("-inf")
        relative_ranges = {}
        for line in fig.data:
            if line_name in line["name"]:
                ax_name = line["yaxis"]
                axis_key = 'yaxis.range'
                if len(ax_name) >= 1:
                    axis_key = 'yaxis{}.range'.format(ax_name[1:])
                ax_min = line["y"].min()
                ax_max = line["y"].max()
                relative_ranges[axis_key] = [ax_min, ax_max]
                if ax_max > fig_max:
                    fig_max = ax_max
                if ax_min < fig_min:
                    fig_min = ax_min
        common_range = {}
        for key in relative_ranges.keys():
            common_range[key] = [fig_min, fig_max]
        return relative_ranges, common_range

    def get_trace_color(self, experiment, fig_name=None):
        if experiment.metadata[CELL_TYPE]  == self.PYR:
            return 'red'
        if experiment.metadata[CELL_TYPE]  == self.IN:
            return 'darkblue'
        else:
            return 'red'
        
    def get_sem_color(self, experiment, fig_name=None):
        if experiment.metadata[CELL_TYPE]  == self.PYR:
            return 'indianred'
        if experiment.metadata[CELL_TYPE]  == self.IN:
            return 'lightblue'    
        else:
            return 'indianred'

    def get_spikes_color(self, experiment, fig_name=None):
        if experiment.metadata[CELL_TYPE]  == self.PYR:
            return 'blue'
        if experiment.metadata[CELL_TYPE]  == self.IN:
            return 'red'
        else:
            return 'blue'
        
    def get_subthreshold_color(self):
        trace_color = 'black'
        sem_color = 'gray'
        return trace_color, sem_color

    def get_bins_slider(self):
        return st.slider('Choose the number of bins:', 40, 200, 40, 20, )
    
    def get_cell_slider(self, experiment):
        if len(experiment.metadata[GOOD_CELLS]) == 1:
            return experiment.metadata[GOOD_CELLS][0]
        return st.select_slider('Choose cell number:',  options=experiment.metadata[GOOD_CELLS])

    def create_images_fig(self, fig_name, experiment):
        if fig_name is None:
            return self._create_image_fig_FOV(experiment)

    def _create_image_fig_FOV(self, experiment):
        volpy_data, slm_patterns, mean_image, _ = pipe_utils.get_pipline_results_data(
            experiment.metadata[CAGE], experiment.metadata[MOUSE_NAME], experiment.metadata[SEQ])
        images_fig = data_utils.display_FOV_images(mean_image, slm_patterns, volpy_data)
        return images_fig

    def create_fig(self, fig_name, experiment):
        if fig_name == CELLS_ACTIVITY:
            return self._create_fig_cells_activity(experiment)
        if fig_name == FR_AND_SUB:
            return self._create_fig_fr_and_sub(experiment)
        if fig_name == LAP_FR:
            return self._create_fig_lap_firing_rate(experiment)
        if fig_name == ACTIVITY_PER_LAP:
            return self._create_fig_activity_per_lap(experiment)
        if fig_name == LONGITUDIAL_ANALYSIS:
            return self._create_fig_longitudinal(experiment)
        if fig_name == SPIKE_DETECTION:
            return self._create_fig_spike_detection(experiment)
           
    def _create_fig_cells_activity(self, experiment):
        def _get_layout_cells_activity(experiment):
            rows_num = experiment.metadata[CELLS_NUM]
            row_heights = [25] * rows_num
            if experiment.behavior_flag:
                rows_num += 1
                row_heights += [5]
            fig = make_subplots \
                (
                rows = rows_num, cols = 1,
                vertical_spacing = 0.01, row_heights = row_heights, shared_xaxes=True, 
                row_titles = [self.CELL_NUMBER_PREFIX + str(i) for i in experiment.metadata[GOOD_CELLS]],
                y_title = self.DELTA_F , x_title = self.AX_TITLE_FRAMES
                )
            if experiment.behavior_flag:
                fig['layout']['yaxis'+str(experiment.metadata[CELLS_NUM]+1)]['title'] = self.AX_TITLE_POSITION
                fig['layout']['yaxis'+str(experiment.metadata[CELLS_NUM]+1)]['title'].update(font=dict(size=10))
                fig.update_yaxes(showticklabels=False, row=experiment.metadata[CELLS_NUM]+1, col=1)
            fig.update_layout(showlegend=True)
            fig.update_layout(title=self.get_generic_title(experiment), title_x=0.45, font=dict(size=18))
            return fig 
        
        def _add_data_cells_acivity(fig, experiment):
            df = experiment.preprocess_traces(experiment.raw_data)
            traces = experiment.get_traces(df)
            spikes_timming = experiment.get_spikes_timming(df)
            spike_heights = experiment.get_spikes_height(spikes_timming, traces)
            trace_color = self.get_trace_color(experiment)
            spike_color = self.get_spikes_color(experiment)
            for i, cell in enumerate(experiment.metadata[GOOD_CELLS]):
                trace = traces[i]
                spikes_times = spikes_timming[i]
                spikes_height = spike_heights[i]
                trace_ax_min, trace_ax_max = self.get_axis_limit(trace, 0.25, 0.25)
                fig.add_scatter(name=self.SPIKES, x=spikes_times, y=spikes_height, mode='markers', legendgroup =2, showlegend=i==0,
                                marker=dict(size=2.5, color=spike_color), visible='legendonly', row=i+1, col=1)
                fig.add_scatter(name=self.CELL +str(cell), x=np.arange(len(trace)), y=trace,
                                line=dict(color=trace_color, width=0.4), showlegend=False, row=i+1, col=1)
                fig.update_yaxes(row=i+1, col=1, range=[trace_ax_min, trace_ax_max])

            if experiment.behavior_flag:
                behave_col = experiment.get_position(df)
                lick_times = experiment.get_licks_timming(df)
                lick_value = 1.01 * behave_col[lick_times]
                fig.add_trace(go.Scatter(x=np.arange(len(behave_col)), y=behave_col, yaxis="y2",
                            showlegend=False, marker=dict(color='black')), row=experiment.metadata[CELLS_NUM]+1, col=1, )
                fig.add_scatter(name = " licks", x=lick_times, y=lick_value, mode='markers', marker=dict(
                    size=2.5, color=spike_color), visible='legendonly', row=experiment.metadata[CELLS_NUM]+1, col=1)
            return fig

        fig = _get_layout_cells_activity(experiment)
        fig = _add_data_cells_acivity(fig, experiment)
        return fig
    
    def _create_fig_fr_and_sub(self, experiment):
        bins_num = self.get_bins_slider()
        def _get_layout_fr_and_sub(experiment, bins_num):
            if experiment.metadata[REMAPPING]:
                cols = 2
                spacing_between_sub_plots = 0.1
                titels = ['<b>' + "FAMILIAR </b>" + "<br>Cell # " + str(experiment.metadata[GOOD_CELLS][0]),
                          '<b>' + "NOVEL </b>" + "<br>Cell # " + str(experiment.metadata[GOOD_CELLS][0])] + \
                        [item for sublist in [["Cell # " + str(i)]*2 for i in experiment.metadata[GOOD_CELLS][1:]] for item in sublist]
            else:
                cols = 1
                spacing_between_sub_plots = 0.05
                titels = [self.CELL_NUMBER_PREFIX + str(i) for i in experiment.metadata[GOOD_CELLS]]
            fig = make_subplots(
                rows = experiment.metadata[CELLS_NUM],
                cols = cols,
                shared_xaxes = True,
                vertical_spacing = spacing_between_sub_plots,
                specs=[[{"secondary_y": True}] * cols] * experiment.metadata[CELLS_NUM],
                subplot_titles=titels
                )

            cm_per_bin = experiment.calculate_bin_length(bins_num, experiment.data)
            fig_title = self.get_generic_title(experiment) 
            fig_title += '<br>Firing rate & subthreshold<br>' + str(cm_per_bin) + self.CM_PER_BIN
            fig.update_yaxes(title_text=self.AX_TITLE_FR, secondary_y=True, row=experiment.metadata[CELLS_NUM], col=cols)
            fig.update_yaxes(title_text=self.DELTA_F, secondary_y=False, row=experiment.metadata[CELLS_NUM], col=1)
            fig.update_xaxes(title_text=self.AX_TITLE_POSITION, row=experiment.metadata[CELLS_NUM])
            fig.update_layout(title=fig_title, title_x=0.45, title_y=0.975)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            if experiment.metadata[CELLS_NUM] > 1:
                fig.update_layout(height=experiment.metadata[CELLS_NUM]*200, width=1000)
            return fig

        def _add_data_fr_and_sub(fig, experiment, bins_num):
            rz_start_bin, rz_end_bin = experiment.get_reward_zone_bins(experiment.data, bins_num)
            if experiment.metadata[REMAPPING]:
                fam_df, nov_df = experiment.get_fam_and_novel_df()
                fam_df_fr = experiment.get_mean_firing_rate(bins_num, fam_df)
                nov_df_fr = experiment.get_mean_firing_rate(bins_num, nov_df)
                fam_df_sub = experiment.get_subthreshold_activity(bins_num, fam_df)
                nov_df_sub = experiment.get_subthreshold_activity(bins_num, nov_df)
                fam_df = fam_df_fr.merge(fam_df_sub, on=BIN, how='left')
                nov_df = nov_df_fr.merge(nov_df_sub, on=BIN, how='left')
                dfs = [fam_df, nov_df]
            else:
                df_fr = experiment.get_mean_firing_rate(bins_num)
                df_sub = experiment.get_subthreshold_activity(bins_num)
                df = df_fr.merge(df_sub, on=BIN, how='left')
                dfs = [df]
            trace_color = self.get_trace_color(experiment)
            sem_color = self.get_sem_color(experiment)
            subthresh_color, subthresh_sem_color = self.get_subthreshold_color()
            cm_per_bin = experiment.calculate_bin_length(bins_num, experiment.data)
            for col_num, df in enumerate(dfs, start=1):
                for row_num, cell_num in enumerate(experiment.metadata[GOOD_CELLS], start=1):
                    vis = True if row_num == 1 and col_num ==1 else False                    
                    fig.add_scatter(name='Firing rate', 
                                    x=df[BIN] * cm_per_bin, y=df[MEAN_FR_PREFIX + str(cell_num)], 
                                    row=row_num, col=col_num, 
                                    legendgroup=0, showlegend=vis, 
                                    line=dict(color=trace_color, width=4), secondary_y=True)
                    
                    fig.add_scatter(name=self.LOW_SEM,  
                                    x=df[BIN] * cm_per_bin,
                                    y=df[MEAN_FR_PREFIX + str(cell_num)]- df[SEM_FR_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=sem_color), secondary_y=True, 
                                    legendgroup=1, showlegend=False, fill=None, row=row_num, col=col_num, visible=True)
                    
                    fig.add_scatter(name='Firing rate SEM', 
                                    x=df[BIN] * cm_per_bin,
                                    y=df[MEAN_FR_PREFIX + str(cell_num)]+ df[SEM_FR_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=sem_color), secondary_y=True, legendgroup=1, 
                                    showlegend=vis, fill="tonexty", row=row_num, col=col_num, visible=True)
                    
                    fig.add_scatter(name="Subthreshold", 
                                    x=df[BIN] * cm_per_bin, y=df[SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                    row=row_num, col=col_num, 
                                    line=dict(color=subthresh_color, width=3), 
                                    secondary_y=False, legendgroup=2, showlegend=vis, visible='legendonly')
                    
                    fig.add_scatter(name=self.LOW_SEM, 
                                      x=df[BIN] * cm_per_bin, 
                                      y=df[SUB_ACTIVITY_PREFIX + str(cell_num)] - df[SEM_SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                      mode='lines', line=dict(color=subthresh_sem_color), secondary_y=False, 
                                      legendgroup=3, showlegend=False, fill=None, row=row_num, col=col_num, visible='legendonly')
                    
                    fig.add_scatter(name='Subthreshold SEM',
                                    x=df[BIN] * cm_per_bin,
                                    y=df[SUB_ACTIVITY_PREFIX + str(cell_num)] + df[SEM_SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=subthresh_sem_color), secondary_y=False, 
                                    legendgroup=3, showlegend=vis, fill="tonexty", row=row_num, col=col_num, visible='legendonly')


                    fig.add_vrect(x0=rz_start_bin * cm_per_bin, x1=rz_end_bin * cm_per_bin, fillcolor="green", opacity=0.25,
                                line_width=0, row=row_num, col=col_num, annotation_font=dict(size=20, color="black"), 
                                annotation_text="<b>RZ</b>", annotation_position="top",)
            
            relative_ranges, common_range = self.get_range_edges_by_line_name(fig, 'Firing rate')
            scale_button = [dict(buttons=[
                    dict(label="relative_scale", method="relayout", args=[relative_ranges]),
                    dict(label="same_scale", method="relayout", args=[common_range])])]
            fig.update_layout(updatemenus=scale_button)

            return fig
        fig = _get_layout_fr_and_sub(experiment, bins_num)
        fig = _add_data_fr_and_sub(fig, experiment, bins_num)
        return fig 
    
    def _create_fig_lap_firing_rate(self, experiment):
        bins_num = self.get_bins_slider()
        def _get_layout_lap_firing_rate(experiment, bins_num):
            fig = make_subplots(
            rows=experiment.metadata[CELLS_NUM], cols=2,
            subplot_titles=tuple(
                [self.CELL_NUMBER_PREFIX + str(i) 
                for i in experiment.metadata[GOOD_CELLS]
                for _ in range(2)
                ]),  
            shared_xaxes=True, 
            vertical_spacing=0.1)
            fig.update_yaxes(title_text="Lap number", col=1)
            fig.update_xaxes(title_text="Position [cm]", row=experiment.metadata[CELLS_NUM])
            cm_per_bin = experiment.calculate_bin_length(bins_num, experiment.data)
            fig_title = self.get_generic_title(experiment) 
            fig_title += '<br>Firing rate per lap (' + str(cm_per_bin) + self.CM_PER_BIN + ')'
            fig.update_layout(title=fig_title, title_x=0.4)
            if experiment.metadata[CELLS_NUM] > 1:
                fig.update_layout(height=experiment.metadata[CELLS_NUM]*200, width=1000)
            else:
                fig.update_layout(height=400, width=1000)
            return fig
        def _add_data_lap_firing_rate(fig, experiment, bins_num):
            cm_per_bin = experiment.calculate_bin_length(bins_num, experiment.data)
            rz_start_bin, rz_end_bin = experiment.get_reward_zone_bins(experiment.data, bins_num)
            df = experiment.get_firing_rate_per_lap(bins_num)
            for i, cell_num in enumerate(experiment.metadata[GOOD_CELLS]):
                fig.add_trace(
                    go.Heatmap(
                    x=df[BIN] * cm_per_bin, 
                    y=df[consts.LAP_COUNTER],
                    z=df[FT_PREFIX + str(cell_num)], 
                    colorscale="amp", showscale=i == 0
                    ), 
                    row=i+1, col=2)
                
                fig.add_trace(
                    go.Heatmap(
                    x=df[BIN] * cm_per_bin, 
                    y=df[consts.LAP_COUNTER],
                    z=(df[FT_PREFIX + str(cell_num)] > 0).astype(int),
                    colorscale="amp", showscale=False
                    ),
                    row=i+1, col=1)

                fig.add_vrect(
                    x0=rz_start_bin * cm_per_bin,
                    x1=rz_end_bin * cm_per_bin, 
                    fillcolor="#c5d9ed", opacity=0.2,
                    line_width=0, row=i+1,  
                    annotation_font=dict(size=20, color="black"), 
                    annotation_text="<b>RZ</b>", 
                    annotation_position="top")

            return fig
        fig = _get_layout_lap_firing_rate(experiment, bins_num)
        fig = _add_data_lap_firing_rate(fig, experiment, bins_num)
        return fig 
    
    def _create_fig_activity_per_lap(self, experiment):
        cell_num = self.get_cell_slider(experiment)
        def _get_layout_activity_per_lap(experiment, cell_num):
            if experiment.metadata[REMAPPING]:
                cols_num = 2
                fam_df, nov_df = experiment.get_fam_and_novel_df()
                laps_num_fam = experiment.get_laps_number(fam_df)
                laps_num_nov = experiment.get_laps_number(nov_df)
                laps_num = max(laps_num_fam, laps_num_nov)
                width_scale = 2
                col_title = ["FAMILIAR", "NOVEL"]
            else:
                cols_num = 1
                laps_num = experiment.get_laps_number(experiment.data)
                width_scale = 1
                col_title = [""]
            fig = make_subplots(
                rows=laps_num, cols=cols_num, 
                shared_xaxes='all', shared_yaxes='all', 
                vertical_spacing=0.0, 
                row_width=[1/laps_num]*laps_num,
                row_titles=["lap #" + str(i+1) for i in range(laps_num)], 
                column_titles=col_title)
            if experiment.metadata[REMAPPING]:
                fig.update_xaxes(title_text=self.AX_TITLE_FRAMES, showticklabels=True, row=max(laps_num_fam, laps_num_nov))
            else:
                fig.update_xaxes(title_text=self.AX_TITLE_FRAMES, row=laps_num, col=1)
            fig.update_yaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=True, row=laps_num//2, col=1)
            fig.update_yaxes(title_text=self.DELTA_F, row=laps_num//2, col=1)
            fig_title = self.get_generic_title(experiment) 
            fig_title += '<br>Activity per lap: Cell # ' + str(cell_num)
            fig.update_layout(title=fig_title, title_x=0.4)
            if experiment.metadata[CELLS_NUM] > 1:
                fig.update_layout(height=experiment.metadata[CELLS_NUM]*500, width=1000)
            else:
                fig.update_layout(height=1000, width=1000*width_scale)
            return fig
        
        def _add_data_activity_per_lap(fig, experiment, cell_num):
            if experiment.metadata[REMAPPING]:
                fam_df, nov_df = experiment.get_fam_and_novel_df()
                dfs = [fam_df, nov_df]
            else:
                dfs = [experiment.data]
            trace_col_prefix = data_utils.get_trace_col_prefix(dfs[0])
            trace_color = self.get_trace_color(experiment)
            for col_num, df in enumerate(dfs, start=1):
                for trace_num, (lap, lap_data) in enumerate(df.groupby([consts.LAP_COUNTER])):
                    trace = lap_data[trace_col_prefix + str(cell_num)]
                    lap_data = lap_data.reset_index()
                    lick_times = experiment.get_licks_timming(lap_data)
                    lick_value = trace.max() * np.ones(len(lick_times))
                    fig.add_scatter(name="env_"+str(col_num) + "_lap_" +str(lap), 
                                    x=np.arange(len(trace)), y=trace,
                                    line=dict(color=trace_color, width=0.4), 
                                    showlegend=False, 
                                    row=trace_num+1, col=col_num,
                                    xaxis='x'+str(col_num))

                    fig.add_scatter(name=" licks",
                                    x=lick_times, y=lick_value,
                                    mode='markers', 
                                    marker=dict(size=2.5, color='black'),
                                    visible='legendonly', 
                                    showlegend=(trace_num == 0) and (col_num == 1), 
                                    legendgroup=0, 
                                    row=trace_num+1, col=col_num, 
                                    xaxis='x'+str(col_num))

            return fig
        fig = _get_layout_activity_per_lap(experiment, cell_num)
        fig = _add_data_activity_per_lap(fig, experiment, cell_num)
        return fig 
    
    def _create_fig_longitudinal(self, experiment):       
        def _get_layout_longitudinal_fr_and_sub(experiment, cell_num):
            long_experiments = self.get_longitudinal_data(experiment, cell_num)
            sessions_num = len(long_experiments)

            remapping = False
            for exp in long_experiments:
                if exp.metadata[REMAPPING]:    
                    remapping = True

            if remapping:
                cols = 3
                column_widths = [0.2, 0.4, 0.4]
                column_titles=["Mean Image", "FAMILIAR", "NOVEL"]

            else:
                cols = 2
                column_widths=[0.3, 0.7]
                column_titles = ["Mean Image", "Firing Rate"]

            
            fig = make_subplots(
                rows=sessions_num, cols=cols, column_widths=column_widths, 
                shared_yaxes='columns', shared_xaxes='columns',
                specs=[[{"secondary_y": True}] * cols] * sessions_num,
                column_titles = column_titles,
                row_titles=["Week # " + str(i) for i in range(1, sessions_num+1)])

            fig.update_yaxes(title_text=self.AX_TITLE_FR, secondary_y=True, row=sessions_num, col=cols)
            fig.update_yaxes(title_text=self.DELTA_F, secondary_y=False, row=sessions_num, col=2)
            fig.update_xaxes(title_text=self.AX_TITLE_POSITION, row=sessions_num)

            fig.update_layout(coloraxis={'colorscale': 'gray'})
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(showticklabels=False,  col=1)
            fig.update_xaxes(showticklabels=False,  col=1)            
            fig_title = experiment.metadata[CAGE] + " " + \
            experiment.metadata[MOUSE_NAME] + " " + \
            experiment.metadata[FOV] + \
            " cell " + str(cell_num) + "<br>" + \
            str(sessions_num) + " Weeks of recordings"
            fig.update_layout(title=fig_title, title_x=0.5)
            for i in range(sessions_num):
                indent = 2
                if remapping:   
                    indent = 3
                fig.layout.annotations[i+indent].update(x=0,
                xshift=-65, textangle=0, font={'size': 12})
            # fig.layout.annotations[sessions_num+2].update(x=0.65)  # x title
            # fig.layout.annotations[sessions_num+3].update(x=0.38)  # y title
            # cancel share x axis on the first column
            fig.update_xaxes(matches=None, col=1)
            # cancel share x axis on the first column
            fig.update_yaxes(matches=None, col=1)
            fig.update_yaxes(autorange="reversed", col=1)
            return fig, long_experiments

        def _get_layout_longitudinal_activity(experiment, cell_num):
            long_experiments = self.get_longitudinal_data(experiment, cell_num)
            sessions_num = len(long_experiments)
            fig = make_subplots(
                rows=sessions_num, cols=2, column_widths=[0.3, 0.7], 
                shared_yaxes='columns', shared_xaxes='columns',
                y_title=self.DELTA_F, x_title="Time [ms]", 
                column_titles=["Mean Image", "Cell Activity"], 
                row_titles=["Week # " + str(i) for i in range(1, sessions_num+1)])
            fig.update_layout(coloraxis={'colorscale': 'gray'})
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(showticklabels=False,  col=1)
            fig.update_xaxes(showticklabels=False,  col=1)            
            fig_title = experiment.metadata[CAGE] + " " + \
            experiment.metadata[MOUSE_NAME] + " " + \
            experiment.metadata[FOV] + \
            " cell " + str(cell_num) + "<br>" + \
            str(sessions_num) + " Weeks of recordings"
            fig.update_layout(title=fig_title, title_x=0.5)
            for i in range(sessions_num):
                fig.layout.annotations[i+2].update(x=0,
                xshift=-65, textangle=0, font={'size': 12})
            fig.layout.annotations[sessions_num+2].update(x=0.65)  # x title
            fig.layout.annotations[sessions_num+3].update(x=0.38)  # y title
            # cancel share x axis on the first column
            fig.update_xaxes(matches=None, col=1)
            # cancel share x axis on the first column
            fig.update_yaxes(matches=None, col=1)
            fig.update_yaxes(autorange="reversed", col=1)
            return fig, long_experiments

        def _add_images_and_contours(fig, long_experiments, cell_num):
            for i, experiment in enumerate(long_experiments):
                
                _, slm_patterns, mean_image, _ = pipe_utils.get_pipline_results_data(
                    experiment.metadata[CAGE], experiment.metadata[MOUSE_NAME], experiment.metadata[SEQ])
                cell_contour = data_utils.get_cell_contour(slm_patterns, cell_num)

                fig_img = px.imshow(mean_image, origin='lower')
                fig.add_trace(fig_img.data[0], row=i+1, col=1)
                fig.add_trace(go.Scatter(
                    name="contour",
                    x=cell_contour[:, 0],
                    y=cell_contour[:, 1], 
                    line=dict(color="LightSkyBlue", width=2),
                    legendgroup=2, showlegend=i == 0,
                    ), row=i+1, col=1)
            return fig

        def _add_data_longitudinal_activity(fig, long_experiments, cell_num):            
            for i, experiment in enumerate(long_experiments):
                df = experiment.preprocess_traces(experiment.raw_data)
                traces = experiment.get_traces(df)
                spikes_timming = experiment.get_spikes_timming(df)
                spike_heights = experiment.get_spikes_height(spikes_timming, traces)
                trace_color = self.get_trace_color(experiment)
                spike_color = self.get_spikes_color(experiment)
                cell_idx = experiment.get_cell_idx(cell_num)
                trace_ax_min, trace_ax_max = self.get_axis_limit(traces[cell_idx], 0.25, 0.25)
                
                fig.add_scatter(name="Week # " +str(i+1), 
                                x=np.arange(len(traces[cell_idx])), 
                                y=traces[cell_idx],
                                line=dict(color=trace_color, width=0.4), 
                                showlegend=False, 
                                row=i+1, col=2)
                fig.add_scatter(name=self.SPIKES, 
                                x=spikes_timming[cell_idx], y=spike_heights[cell_idx], 
                                mode='markers', legendgroup =2, 
                                showlegend=i==0, visible='legendonly',
                                marker=dict(size=2.5, color=spike_color), 
                                row=i+1, col=2)
                fig.update_yaxes(row=i+1, col=2, range=[trace_ax_min, trace_ax_max])
            return fig
        
        def _add_data_longitudinal_fr_and_sub(fig, long_experiments, cell_num, bins_num):
            for row_num, experiment in enumerate(long_experiments, start=1):
                rz_start_bin, rz_end_bin = experiment.get_reward_zone_bins(experiment.data, bins_num)
                if experiment.metadata[REMAPPING]:
                    fam_df, nov_df = experiment.get_fam_and_novel_df()
                    fam_df_fr = experiment.get_mean_firing_rate(bins_num, fam_df)
                    nov_df_fr = experiment.get_mean_firing_rate(bins_num, nov_df)
                    fam_df_sub = experiment.get_subthreshold_activity(bins_num, fam_df)
                    nov_df_sub = experiment.get_subthreshold_activity(bins_num, nov_df)
                    fam_df = fam_df_fr.merge(fam_df_sub, on=BIN, how='left')
                    nov_df = nov_df_fr.merge(nov_df_sub, on=BIN, how='left')
                    dfs = [fam_df, nov_df]
                else:
                    df_fr = experiment.get_mean_firing_rate(bins_num)
                    df_sub = experiment.get_subthreshold_activity(bins_num)
                    df = df_fr.merge(df_sub, on=BIN, how='left')
                    dfs = [df]
                trace_color = self.get_trace_color(experiment)
                sem_color = self.get_sem_color(experiment)
                subthresh_color, subthresh_sem_color = self.get_subthreshold_color()
                cm_per_bin = experiment.calculate_bin_length(bins_num, experiment.data)
                for col_num, df in enumerate(dfs, start=2):
                    vis = True if row_num == 1 and col_num == 2 else False                    
                    fig.add_scatter(name='Firing rate', 
                                    x=df[BIN] * cm_per_bin, y=df[MEAN_FR_PREFIX + str(cell_num)], 
                                    row=row_num, col=col_num, 
                                    legendgroup=0, showlegend=vis, 
                                    line=dict(color=trace_color, width=4), secondary_y=True)
                    
                    fig.add_scatter(name=self.LOW_SEM,  
                                    x=df[BIN] * cm_per_bin,
                                    y=df[MEAN_FR_PREFIX + str(cell_num)]- df[SEM_FR_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=sem_color), secondary_y=True, 
                                    legendgroup=1, showlegend=False, fill=None, row=row_num, col=col_num, visible=True)
                    
                    fig.add_scatter(name='Firing rate SEM', 
                                    x=df[BIN] * cm_per_bin,
                                    y=df[MEAN_FR_PREFIX + str(cell_num)]+ df[SEM_FR_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=sem_color), secondary_y=True, legendgroup=1, 
                                    showlegend=vis, fill="tonexty", row=row_num, col=col_num, visible=True)
                    
                    fig.add_scatter(name="Subthreshold", 
                                    x=df[BIN] * cm_per_bin, y=df[SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                    row=row_num, col=col_num, 
                                    line=dict(color=subthresh_color, width=3), 
                                    secondary_y=False, legendgroup=2, showlegend=vis, visible='legendonly')
                    
                    fig.add_scatter(name=self.LOW_SEM, 
                                        x=df[BIN] * cm_per_bin, 
                                        y=df[SUB_ACTIVITY_PREFIX + str(cell_num)] - df[SEM_SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                        mode='lines', line=dict(color=subthresh_sem_color), secondary_y=False, 
                                        legendgroup=3, showlegend=False, fill=None, row=row_num, col=col_num, visible='legendonly')
                    
                    fig.add_scatter(name='Subthreshold SEM',
                                    x=df[BIN] * cm_per_bin,
                                    y=df[SUB_ACTIVITY_PREFIX + str(cell_num)] + df[SEM_SUB_ACTIVITY_PREFIX + str(cell_num)], 
                                    mode='lines', line=dict(color=subthresh_sem_color), secondary_y=False, 
                                    legendgroup=3, showlegend=vis, fill="tonexty", row=row_num, col=col_num, visible='legendonly')


                    fig.add_vrect(x0=rz_start_bin * cm_per_bin, x1=rz_end_bin * cm_per_bin, fillcolor="green", opacity=0.25,
                                line_width=0, row=row_num, col=col_num, annotation_font=dict(size=20, color="black"), 
                                annotation_text="<b>RZ</b>", annotation_position="top",)
            
                relative_ranges, common_range = self.get_range_edges_by_line_name(fig, 'Firing rate')
                scale_button = [dict(buttons=[
                        dict(label="relative_scale", method="relayout", args=[relative_ranges]),
                        dict(label="same_scale", method="relayout", args=[common_range])])]
                fig.update_layout(updatemenus=scale_button)
            return fig

        plot_type = st.radio("Chosse a plot:", ('Activity', 'Firing rate'))
        cell_num = self.get_cell_slider(experiment)
        if plot_type == "Activity":
            fig, long_experiments = _get_layout_longitudinal_activity(experiment, cell_num)
            fig = _add_images_and_contours(fig, long_experiments, cell_num)
            fig = _add_data_longitudinal_activity(fig, long_experiments, cell_num)
        if plot_type == "Firing rate":
            bins_num = self.get_bins_slider()
            fig, long_experiments = _get_layout_longitudinal_fr_and_sub(experiment, cell_num)
            fig = _add_images_and_contours(fig, long_experiments, cell_num)
            fig = _add_data_longitudinal_fr_and_sub(fig, long_experiments, cell_num, bins_num)
        return fig 

    def _create_fig_spike_detection(self, experiment):
        import spike_detection_from_db as sd
        data_file = os.path.join(paths.DATASET_DIR_WINDOWS, experiment.metadata[CAGE], experiment.metadata[MOUSE_NAME], experiment.metadata[SEQ] + '.parquet')
        data_df = pd.read_parquet(data_file)
        fig = sd.run_spike_detection(experiment.metadata[CAGE], experiment.metadata[MOUSE_NAME], 
                                     experiment.metadata[SEQ], experiment.metadata[GOOD_CELLS], 
                                     experiment.metadata[CELL_TYPE], experiment.metadata[FOV], experiment.metadata[FRAME_RATE], data_df)
        return fig


    def _create_fig_template(self, experiment):
        bins_num = self.get_bins_slider()
        def _get_layout_template(experiment, bins_num):
            pass
            return fig
        def _add_data_template(fig, experiment, bins_num):
            pass
            return fig
        fig = _get_layout_template(experiment, bins_num)
        fig = _add_data_template(fig, experiment, bins_num)
        return fig 


