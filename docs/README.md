# FromExperimentToAnalysis

Our goal is to bridge the gap between the end of the experiment, a time point in a lab project where the experimentalist 
completed the data acquisition from the experiment setup but still didn't started to analyze it, to the analysis step. 
Many times, some preprocessing steps need to be done before one can actually start to analyze its data.

In our case, we finished the experiment with two main files - a video from mice brain 
(with corresponding metadata files like SLM patterns coordinates; 
xml file with experiment details (e.g. width and height of the fov)), 
and behavioral data with the animal position during the imaging.

To start with the analysis step we would like to:
1. extract voltage traces per each recorded cell
2. align the traces with mice behavior in each camera frame
3. save information about the different experiment conditions

Therefore our pipeline will be composed from three pillars:
1. Image processing algorithms that extract the voltage signal of the recorded cells from the videos and reduce the inherent noise as much as possible.
2. Data merger that align the behavioral and the resulting traces and merge them to one file containing detailed information about the mice behavior and the corresponding brain activity in each time point.
3. DataBase of all our experiment that enable us to study series of experiment and filter experiment with respect to common studied features.

# Getting started 

### 1. clone the repository:

```git clone https://github.com/yoavadamlab/FromExperimentToAnalysis.git```

### 2. open `utils/files_paths.py` in a text editor and change the paths to your locals DB files

### 3. Create conda environment:

```
conda env create -f environment.yml
conda activate voltage_imaging_pipeline
streamlit run source\gui.py
```

for detailed explanation about the code structure open `code_scheme.excalidraw` in https://excalidraw.com

### Adam Lab users:

### Folders structure for the pipeline:

```
microscope_session\
├── FOV_number\
│   ├── experiment1_setup\
│   │   ├── Image_001_001.raw
│   │   ├── ROIs.xaml
│   │   ├── Experiment.xml
│   │   ├── TS\
│   │   │	├── ThorRealTimeDataSettings.xml
│   │   │	└── Episode_0000.h5
│   │── experiment2_setup\
│	.
│	.
│	.
```
### After successful pipeline you should get the following files:
```
microscope_session\
├── FOV_number\
│   ├── experiment1_setup\
│   │   ├── Image_001_001.raw
│   │   ├── ROIs.xaml
│   │   ├── Experiment.xml
│   │   ├── TS\
│   │   │	├── ThorRealTimeDataSettings.xml
│   │   │	└── Episode_0000.h5
│   │   ├── traces\
│   │   │	└── timestamp_traces.csv --> raw traces 
│   │   ├── Traces_ROIs\
│   │   │	└── timestamp.fig --> first glance matlab figure
│   │   ├── pipeline_params\
│   │   │	└── timestamp_params.json --> the parameters that were used by the pipeline
│   │   ├── pipeline_results\
│   │      ├── motion_corrected\
│   │	   │	├── timestamp_motion_corrected.tif --> motion corrected video
│   │	   │	├── timestamp_mc_and_raw.tif --> raw and mc videos side by side
│   │	   │	├── shifts\
│   │	   │	├──	└── timestamp_mc_shifts.mat --> motion corrected shifts per each frame 
│   │	   │	├── traces\
│   │	   │	│	└── timestamp_traces.csv --> motion corrected traces 
│   │	   │	├── spatial_footprint_traces\
│   │	   │	│	└── timestamp_traces.csv --> the final traces of the pipeline 
│   │	   ├── volpy_data\
│   │	   │	└── timestamp\
│   │      │          ├── cell_number_roi_data.png --> images of volpy outputs
│   │	   │          ├── mean_image.npy --> mean image of the video
│   │	   │          ├── slm_patterns.npy --> numpy object of the SLM patterns
│   │	   │          └── volpy_data.npy --> numpy object with all the volpy outputs
│   │── experiment2_setup\
│	.
│	.
│	.
```
