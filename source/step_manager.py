import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths
from utils import credentials as cred
from utils import pipeline_utils as pipe_utils
import json
import string
import datetime as dt
import pandas as pd


class PipelinesManager:
    def __init__(self, session_time):
        self.pipe_counter = -1
        self.pipelines = {}
        self.running_pipelines = []
        self.finished_pipelines = []
        self.failed_pipelines = []
        self.cancelld_pipelines = []
        self.gui_session_time = session_time
        self.param_files = []

    def fetch_new_pipelines(self):
        session_logs_dir = os.path.join(paths.PIPELINE_LOGS_DIR, self.gui_session_time)
        for name in os.listdir(session_logs_dir):
            if os.path.isdir(os.path.join(session_logs_dir, name)):
                if name not in self.param_files:
                    self.param_files.append(name)
                    param_file = os.path.join(session_logs_dir, name, consts.PARAMS_FILE_SUFFIX_NAME[1:])
                    with open(param_file) as json_file:
                        gui_params = json.load(json_file)
                        self.create_new_pipeline(gui_params)
        return

    def create_new_pipeline(self, gui_params):
        self.pipe_counter += 1
        pipe = Pipeline(gui_params, self.pipe_counter)
        self.pipelines[self.pipe_counter] = pipe
        self.running_pipelines.append(self.pipe_counter)

    def manage_pipelines_steps(self):
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            if pipe.state == consts.WAITING:
                pipe.run_queued_step()
            elif pipe.state == consts.EXECUTING:
                pipe.manage_running_step()
            elif pipe.state == consts.FAILED:
                self.failed_pipelines.append(pipe_num)
            elif pipe.state == consts.FINISHED:
                self.finished_pipelines.append(pipe_num)
            elif pipe.state == consts.CANCELLD:
                self.cancelld_pipelines.append(pipe_num)
        return 


    def save_pipelines_logs(self):
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            logs = pipe.get_step_info() # (pipe.current_step.step_name, pipe.current_step.state, pipe.current_step.logs)
            pipe_log_dir =  os.path.join(paths.PIPELINE_LOGS_DIR, self.gui_session_time, pipe.log_dir)
            if logs is not None:
                step_log_file = os.path.join(pipe_log_dir, logs[0] + '.txt')
                with open(step_log_file, 'w') as f:
                    f.write(f"Job state: {logs[1]}\n")
                    if logs[2] is not None:
                        for line in logs[2]:
                            f.write(f"{line}\n")
        return 

    def fetch_logs(self):
        logs = []
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            l = pipe.current_step.get_logs()
            logs.append(l)
            # send it to gui for some actions
        return logs
        
    def fetch_step_states(self):
        states = [] # this list just for tests 
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            s = pipe.current_step.get_state()
            states.append(s)
            # send it to gui for some actions
        return states
    
    def pipeline_termination(self):
        """
        handle with crashe pipelines (should stop the pipeline and not continue to send more jobs from it)
        also should handle completion of a pipeline by telling it to the gui
        """
        for pipe_num in self.finished_pipelines + self.failed_pipelines + self.cancelld_pipelines:
            if pipe_num in self.running_pipelines:
                if pipe_num in self.finished_pipelines:
                    self.save_experiment_details_to_upload_table(pipe_num)
                self.running_pipelines.remove(pipe_num)
        # self.finished_pipelines
        # self.failed_pipelines = []
        # self.cancelld_pipelines = []
    
    def save_experiment_details_to_upload_table(self, pipe_num):
        def get_files_suffix(cage, mouse_name, seq):
            mouse_path = os.path.join(paths.DATASET_DIR, cage, mouse_name)
            suffixes = [""] + ["_" + l for l in list(string.ascii_lowercase)]
            for suf in reversed(suffixes):
                if os.path.isfile(os.path.join(mouse_path, seq + suf + '.csv')):
                    break # now suf = the right suffix for the file name
            return seq + suf
                
        pipe = self.pipelines[pipe_num]
        cage = pipe.gui_params[consts.CAGE]
        mouse_name = pipe.gui_params[consts.MOUSE_NAME]
        seq = get_files_suffix(cage, mouse_name, pipe.gui_params[consts.SEQ])
        cell_type = pipe.gui_params[consts.CELL_TYPE]
        video_path = pipe.gui_params[consts.RAW_VIDEO_PATH]
        experiment_date = dt.datetime.fromtimestamp(os.path.getctime(video_path))
        experiment_date_str = experiment_date.strftime("%Y-%m-%d")
        pipe_utils.save_record_to_DB_queue(experiment_date_str, cage, mouse_name, seq, cell_type, video_path)


class Pipeline:
    def __init__(self, gui_params, pipe_counter):
        steps_lst = self._get_pipeline_steps_from_gui(gui_params)
        self.param_path = self._save_params_to_json(gui_params)
        self.pipeline_steps = self.parse_pipeline_steps(steps_lst)
        self.state = consts.WAITING
        self.current_step = None
        self.completed_steps = []
        self.serial_num = pipe_counter
        self.log_dir = self._get_log_dir(gui_params)
        self.gui_params = gui_params

    def _get_log_dir(self, gui_params):
        pipe_dir = "_".join([
                            gui_params[consts.CAGE], gui_params[consts.MOUSE_NAME],
                            gui_params[consts.SEQ], gui_params[consts.GUI_TIME]]) 
        return pipe_dir

        
    def _get_pipeline_steps_from_gui(self, gui_params):
        """
        return a list of tupels containing the different steps
        from GUI and a flag indicate if they need to be run:
        # [(step_name, run_flag) e.g. (MC, True)]  the order is matters!
        """
        steps_names = pipe_utils.get_steps_lst()
        steps_lst = []
        for step in steps_names:
            if gui_params[step]:
                steps_lst.append((step, True))
        return steps_lst

    def _save_params_to_json(self, gui_params):
        home_dir = gui_params[consts.HOME_DIR]
        param_dir = os.path.join(home_dir, consts.PARAMS_DIR_NAME)
        pipe_utils.mkdir(param_dir)
        param_file_path = os.path.join(param_dir, gui_params[consts.GUI_TIME] + consts.PARAMS_FILE_SUFFIX_NAME)
        with open(param_file_path , 'w') as fp:
            json.dump(gui_params, fp, indent=4)
        param_file_for_cluster = pipe_utils.windows_to_linux_path(param_file_path)
        return param_file_for_cluster

    def parse_pipeline_steps(self, steps_lst):
        """
        return list of steps to sequentially run for this pipeline
        """
        pipeline_steps = [PipelineStep(step_name, self.param_path)
                          for step_name, run_step in steps_lst
                          if run_step]
        return pipeline_steps

    def run_queued_step(self): 
        self.current_step =  self.pipeline_steps.pop(0)
        self.current_step.run()
        self.state = consts.EXECUTING
    
    def manage_running_step(self):
        if self.current_step.get_state() == consts.JOB_FAILED:
            self.state = consts.FAILED # the manager will crash the pipeline 
        if self.current_step.get_state() == consts.JOB_CANCELLD:
            self.state = consts.CANCELLD # the manager will crash the pipeline 
        if self.current_step.get_state() == consts.JOB_FINISHED:
            self.completed_steps.append(self.current_step)
            self.current_step = None
            if not self.pipeline_steps: # if current step was the last
                self.state = consts.FINISHED
            else:
                self.state = consts.WAITING
    
    def get_step_info(self):
        if self.current_step is not None:
            return (self.current_step.step_name, self.current_step.state, self.current_step.logs)
        else:
            if len(self.completed_steps) > 0:
                return (self.completed_steps[-1].step_name, self.completed_steps[-1].state, self.completed_steps[-1].logs)
            else:
                return None

class PipelineStep:
    """
    represent a single step of the pipeline.
    the step logic implemented in different py files - one per step.
    """

    def __init__(self, step_name, param_path):
        self.step_name = step_name
        self.params = param_path
        self.state = consts.NOT_STARTED
        self.cluister_script = self.get_cluster_script()
        self.cluster_job = ClusterJob(self.cluister_script, self.params)
        self.logs = None
            
    def get_cluster_script(self):
        if self.step_name == consts.RAW_TRACES_EXTRACTION:
            return paths.CLUSTER_RUNNERS_DIR + paths.RAW_TRACES_EXTRACTION_BASH + " " + self.params
        if self.step_name == consts.MOTION_CORRECTION:
            return os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.MOTION_CORRECTION_BASH)
        if self.step_name == consts.DENOISING:
            return os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.DENOISER_BASH)
        if self.step_name == consts.SPATIAL_FOOTPRINT:
            return os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.SPATIAL_FOOTPRINT_BASH)
        if self.step_name == consts.BEHAVIOR_AND_TRACES_MERGE:
            return os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.BEHAVEIOR_TRACES_MERGER_BASH)
        if self.step_name == consts.SPIKE_DETECTION:
            return os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.SPIKE_DETECTION_BASH)


    def run(self):
        self.cluster_job.run_job()

    def get_state(self):
        self._update_state()
        self.get_logs()
        return self.state

    def _update_state(self):
        self.state = self.cluster_job.update_state()

    def cancel_step(self):
        self.cluster_job.cancel_job()

    def get_logs(self):
        logs = self.cluster_job.get_job_logs()
        self.logs = logs
        return logs

class ClusterJob:
    def __init__(self, cluster_script, script_params=""):
        self.script = cluster_script
        self.params = script_params
        self.job_id = None
        self.log_file = None
        self.run_job_command = consts.RUN_JOB_COMMAND
        self.job_state_command = consts.JOB_STATE_COMMAND
        self.cancel_job_command = consts.CANCEL_JOB_COMMAND
        self.get_log_path_command = consts.GET_LOG_PATH_COMMAND

    def run_job(self):
        ssh = SSH_connection()
        command = " ".join([self.run_job_command, self.script, self.params])
        output = ssh.run_command(command) 
        self.job_id = [s for s in output[0].split() if s.isdigit()][0]
        ssh.close()

    def update_state(self):
        ssh = SSH_connection()
        command = self.job_state_command.format(self.job_id)
        output = ssh.run_command(command)
        ssh.close()
        state = self._parse_state(output)
        return state

    def _parse_state(self, output):
        if output == []:
            state = consts.NOT_STARTED
        elif output[0].strip().split(' ')[0] == consts.SLURM_PENDING:
            state = consts.JOB_PENDING
        elif output[0].strip().split(' ')[0] == consts.SLURM_RUNNING:
            state = consts.JOB_RUNNING
        elif output[0].strip().split(' ')[0] == consts.SLURM_FAILED:
            state = consts.JOB_FAILED
        elif output[0].strip().split(' ')[0] == consts.SLURM_FINISHED:
            state = consts.JOB_FINISHED
        elif output[0].strip().split(' ')[0] == consts.SLURM_CANCELLD:
            state = consts.JOB_CANCELLD
        else:
            print("not handel state")
            print(output)
        return state

    def cancel_job(self):
        ssh = SSH_connection()
        command = self.cancel_job_command.format(self.job_id)
        ssh.run_command(command)
        ssh.close()
        self.update_state()
    
    def _set_log_file(self):
        ssh = SSH_connection()
        command = self.get_log_path_command.format(self.job_id)
        output = ssh.run_command(command)
        ssh.close()
        self.log_file =  output[0].strip().split('=')[1]

    def get_job_logs(self):
        if not self.log_file:
            self._set_log_file()
        ssh = SSH_connection()
        logs = ssh.run_command("less " + self.log_file)
        ssh.close()
        logs = [row.strip() for row in logs]
        return logs

class SSH_connection:

    def __init__(self):
        self.host = cred.CLUSTER_HOST
        self.port = cred.CLUSTER_PORT
        self.username = cred.SSH_USERNAME
        self.password = cred.SSH_PASSWORD
        self.timeout = 600 # [seconds]
        self.ssh = self._connect()
    
    def _connect(self):
        import paramiko 
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, self.port, self.username, self.password, timeout=self.timeout)
        return ssh 
    
    def run_command(self, command):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.readlines()
        if output == []:
            output = stderr.readlines()
        return output
    
    def close(self):
        self.ssh.close()
  
def main(args):
    session_time = args[1]
    manager = PipelinesManager(session_time)
    while True:
        manager.fetch_new_pipelines()
        manager.manage_pipelines_steps()
        manager.pipeline_termination()
        manager.save_pipelines_logs()
        time.sleep(10) # check for updates every 10 seconds

if __name__ == "__main__":
    main(sys.argv)

