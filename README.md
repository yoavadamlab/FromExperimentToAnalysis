# From Experiment to Analysis
An end to end Neuroimaging pipeline for video motion correction, trace extraction, spike detection (all adapted from [Volpy](https://github.com/flatironinstitute/CaImAn/tree/52ab5fbfbbcda12a3cacfd319216be5373e6398e/caiman/source_extraction/volpy) with some modifications), followed by data curation and visualization.


<img src= "https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/10b3a580-754d-4e41-913b-c1439159b1d2" width="500">

### 1. Run pipeline on your cluster and monitor its progress

By a simple API you can add pipeline steps and configure your cluster preferences.  
<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/211a1014-f5a9-4862-84d3-6f5d45faabef" width="500">
<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/dc80240d-3721-4e6a-b10f-b013c2834024" width="500">

### 2. Validate pipeline steps
Explore the resulting output from the pipeline for validation of hyper-parameters choice.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/d2445668-a2b7-4dc1-89c0-e83bc89f3210" width="500">

### 3. Spike Detection refinement
Fine tune the detection method and threshold according to the SNR of different trace segments.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/feacc106-3f00-4aa9-90a3-e9782c18707c" width="500">

### 4. Imaging DB
All your data in one place for managing and exploration.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/77c64b17-9452-4508-b99c-46bc7db96a0e" width="500">

### 5. Data visualization
Explore your data in first glance with automated plots.

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/f3831d28-31b1-4ba8-946c-b4733225b11f" width="500">

<img src="https://github.com/yoavadamlab/FromExperimentToAnalysis/assets/98536980/dfb1c50a-a09d-4c2d-a747-dcd67c5df36b" width="500">

## Getting started 

```
git clone https://github.com/yoavadamlab/FromExperimentToAnalysis.git
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
