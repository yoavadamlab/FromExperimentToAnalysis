# From Experiment to Analysis
An end to end Neuroimaging pipeline for video motion correction, trace extraction, spike detection (all adapted from [Volpy](https://github.com/flatironinstitute/CaImAn/tree/52ab5fbfbbcda12a3cacfd319216be5373e6398e/caiman/source_extraction/volpy) with some modifications), followed by data curation and visualization.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/137a5043-5734-45e0-9742-f3a30544c6bb" width="500">

### 1. Run pipeline on your cluster and monitor its progress

By a simple API you can add pipeline steps and configure your cluster preferences.  

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/092c7a5b-62e4-4d2e-b6f9-56bb569a2de6" width="300">
<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/5956317a-a271-4ef7-9f7d-2d22999a9b60" width="300">

### 2. Validate pipeline steps
Explore the resulting output from the pipeline for validation of hyper-parameters choice.

### 3. Spike Detection refinement
Fine tune the detection method and threshold according to the SNR of different trace segments.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/054a73e8-79d6-439a-a806-1e755cc50a28" width="300">

### 4. Imaging DB
All your data in one place for managing and exploration.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/d46409ad-ad2d-4e9c-8b6d-a3770588bc12" width="300">

### 5. Data visualization
Explore your data in first glance with automated plots.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/56c5f634-9bb8-4e5a-93f0-946c34397237" width="300">
<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/5c421ca2-6583-4984-bd33-a14515eeee4a" width="300">


## Getting started 

```
git clone
```
in `utils\file_paths` change paths to your local directories
for more detailed documentation see docs (link)
Now create conda environment and run a streamlit app:
```
conda env create -f environment.yml
conda activate voltage_imaging_pipeline
streamlit run source\gui.py
```

For detailed code structere and how to modify the code for your lab needs, open the `docs/code_scheme.excalidraw` file  in https://excalidraw.com.
