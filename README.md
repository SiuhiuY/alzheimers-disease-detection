## Project Title
Alzheimer's Disease Detection Using EEG Topographic Images

---

## Project Overview
The project explores feature representation for EEG-based Alzheimer's Disease detection, comparing **1D raw EEG signals** with **2D topographic image-like representations** derived from EEG signals. Both representations are evaluated using CNN-based architectures.

The central question is whether representing EEG signals with **explicit spatial mapping** improves classification performance. To address this, the repository implements:
- EEG signal-to-image feature contruction
- Nested Leave-One-Subject-Out cross-validation
- Model comparison

## Problem Statement
Alzheimer's Disease (AD) is a major neurological disorders affecting over 57 million people worldwide. Current diagnostic methods — such as MRI, PET scans, cerebrospinal fluid biomarkers — are costly, invasive, or not widely accessible. EEG offers a non-invasive, affordable, and widely available alternative. However, no formally recognised EEG biomarkers for AD detection currently exist.

The project focuses on the following challenges:
- **Spatial nature of AD pathology:** AD disrupts inter-neuron connections in spatially distributed patterns, suggesting spatial EEG features may contain useful diagnostic information.
- **Uncertainty in feature representation:** Deep learning is widely used in EEG-based AD detection due to its ability to automatically extract features. However, the lack of large-scale public datasets makes it difficult to develop specialised models. As a result, EEG signals are often transformed to align with architecures from more mature fields, such as computer vision and natural langugage processing. The optimal feature representation for deep learning models remains unclear.
- **Limited data and generalisability:** Clinical EEG datasets are typically small, making it difficult to train robust models and raises concerns about model's generalisation.

## Methods


## Repository Structure
```
THESIS_CODES/
│
├── data/                                # data folder
|
├── docs/                                # Flowcharts
│
├── notebooks/                           # EDA, feature analysis and model comparison
│
├── scripts/
│   └── baseline.sh                      # GPU job submission scripts
│
├── src/
│   ├── models/                          # Model files directory
│   ├── calculate_results.py             # Compute evaluation metrics
│   ├── callback.py                      # Training callbacks (early stopping)
│   ├── cross_validation.py              # Cross-validation logic
│   ├── dataset.py                       # Custom PyTorch dataset
│   ├── eeg_processor.py                 # Transform signals into images (single subject)
│   ├── feature_loader.py                # Load image feature and synchronise with labels
│   ├── model_trainer.py                 # Train, evaluate and predict
│   ├── model_tuner.py                   # Hyperparameter tuning
│   ├── subject_processor.py             # Image transformation (all subjects)
│   ├── util.py                          # Helper functions
│   └── __init__.py
│
├── tests/
│   ├── test_processor.py                # Test eeg_processor.py 
│   ├── test_subject.py                  # Test subject_processor.py
│   └── __init__.py
│
├── eegnet_baseline.py                   # 1D EEG signal feature training pipeline
├── experiment.py                        # Single train/test split for quick model behaviour check
├── main.py                              # 2D image feature training pipeline
├── environment.gpu.yml                  # Environment setting for GPU
├── environment.local.yml                # Environment setting for CPU 
├── .gitignore
├── .gitlab-ci.yml
├── README.md
└── requirements.txt

```

## Dataset
Dataset is publicly available on OpenNeuro: https://openneuro.org/datasets/ds004504/versions/1.0.7. 

Resting-state eyes-closed EEG recordings with 19 channels at 500Hz sampling rate. The subject groups include Alzheimer's Disease (AD), Frontaltemporal Dementia (FTD), and healthy controls (CN). 

## Results

### Summary

- EEGNet with 1D features consistently outperformed other models, with an AUROC of 0.82 in Alzheimer's Disease classification, balanced sensitivity and specificity, as well as unbiased prediction between male and female subjects. 
- Frontotemporal dementia proved a challenging dementia type for all models. 
- Frenquency band analysis revealed Delta and Alpha bands to be the most informative for Azheimer's Disease; no single band yielded strong results for Frontotemporal dementia. 
- Subject-level analysis showed considerable inter-subject variability. 
- SHAP-based explainability revealed discriminative spatial patterns for Alzheimer's Disease but not for Frontotemporal dementia. 



## Usage

### Dataset Download 
Download the dataset at https://openneuro.org/datasets/ds004504/versions/1.0.7/download. 

Choose a root directory when prompted and create a subfolder named ```data```. 

### Image Extraction 
To extract images from the EEG signals, run:
```
python -m src.subject_processor
```
The default configurations, including frequency band and the sliding window size, can be adjusted by changing the arguments when calling the methods of ```SubjectProcessor``` class.
```
processor.choose_band(band_name="alpha")
processor.choose_window_size(window_size=4)
```

The images will be saved under a folder name which corresponds to the band name of your choice under ```/data/features/```. 

By implementing the above two steps, your data repository will be arranged as below: 

```
├── data/
│   ├── derivatives/                      # Preprocessed EEG signals directory
│   │   ├── sub-001/
│   │   │   └── eeg/
│   │   │       └── sub-001_task-eyesclosed_eeg.set
│   │   ├── sub-002/
│   │   ├── sub-003/
│   │   ├── ...
│   │   ├── sub-086/
│   │   ├── sub-087/
│   │   └── sub-088/
│   │
│   ├── features/                        # 2D image features directory
│   │   ├── alpha/
│   │   │       └── sub-001_alpha_psd.npy
│   │   └── ...
│   │
│   ├── raw/                             # Unprocessed EEG signals directory
│   │   ├── sub-001/
│   │   │   └── eeg/
│   │   │       ├── sub-001_task-eyesclosed_channels.tsv
│   │   │       ├── sub-001_task-eyesclosed_eeg.json
│   │   │       └── sub-001_task-eyesclosed_eeg.set
│   │   ├── sub-002/
│   │   ├── sub-003/
│   │   ├── ...
│   │   ├── sub-086/
│   │   ├── sub-087/
│   │   └── sub-088/
│   │
│   ├── CHANGES
│   ├── dataset_description.json        
│   ├── participants.json                # Meta data mapping dictionary
│   └── participants.tsv                 # Meta data by each subject
│   └── README                           # Dataset description
```

To test for image extraction steps in ```eeg_processor.py``` and ```subject_processor.py```, run: 
```
pytest -v
```

### Run Experiment

#### 1D feature
1D CNN model pipeline is implemented separately because the original EEGNet architecture is written in Keras. To run signal extraction and nested cross validation:
```
python eegnet_baseline.py
```

#### 2D feature
```
python main.py
```

## Future Direction


## References
