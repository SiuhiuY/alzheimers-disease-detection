# Master Thesis - Data Science & Society

## Project Title
Alzheimer's Disease Detection Using EEG Topographic Images

---

## Project Overview
The project explores feature representation in deep learning with EEG data, comparing 1D raw signal features with topographic image-like 2D features. Both features 

## Repository Structure
```
THESIS_CODES/
│
├── data/
│   ├── derivatives/                     # Preprocessed EEG signals directory
│   ├── features/                        # 2D image features directory
│   ├── raw/                             # Unprocessed EEG signals directory
│   ├── CHANGES
│   ├── dataset_description.json           
│   ├── participants.json                # Meta data mapping dictionary
│   ├── participants.tsv                 # Meta data by each subject
│   └── README                           # Dataset description
│
├── jobs/
│   └── eeg_prep_gpu.sh                  # GPU job submission scripts
│
├── notebooks/                  
│   ├── eda.ipynb                        # Explore EEG signals with MNE library
│   ├── image.ipynb                      # Present the image feature
│   └── results.ipynb                    # Post-training analysis
│
├── src/
│   ├── models/                          # Model files directory
│   ├── dataset.py                       # Custom PyTorch dataset generator 
│   ├── eeg_processor.py                 # Transform signals into images for a single subject
│   ├── feature_loader.py                # Load image feature and synchronise with labels
│   ├── model_trainer.py                 # Train, evaluate and predict
│   ├── model_tuner.py                   # Hyperparameter tuning
│   ├── subject_processor.py             # Image transformation for all subjects
│   ├── util.py                          # Helper functions
│   └── __init__.py
│
├── tests/
│   ├── test_processor.py                # Test eeg_processor.py 
│   ├── test_subject.py                  # Test subject_processor.py
│   └── __init__.py
│
├── cross_validation.py                  # Nested cross validation
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

https://openneuro.org/datasets/ds004504/versions/1.0.7

## Usage

### Dataset Download 
Download the dataset at https://openneuro.org/datasets/ds004504/versions/1.0.7/download

By choosing your root directory, the dataset will automatically be saved in a "data" folder.

### Image Extraction 
To extract images from the EEG signals, run 
```
python -m src.subject_processor
```
The default configurations, including frequency band and the sliding window size, can be adjusted by changing the arguments when calling the methods.
```
processor.choose_band(band_name="alpha")
processor.choose_window_size(window_size=4)
```

The images will be saved under a folder name which which corresponds to the band name of your choice in ```/data/features/```. 

By implementing the above two steps, your data repository will be arranged as below: 

```
├── data/
│   ├── derivatives/
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
│   ├── features/
│   │   ├── alpha/
│   │   │       └── sub-001_alpha_psd.npy
│   │   └── ...
│   │
│   ├── raw/
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
│   ├── participants.json
│   ├── participants.tsv
│   └── CHANGES
```

To test for image extraction steps in eeg_processor.py and subject_processor.py, run
```
pytest -v
```

### Run Experiment


## Requirements
```bash
python==3.10
torch==2.1.0
torchmetrics==1.2.0
numpy==1.26.4
mne==1.7.0
scikit-learn==1.3.0
matplotlib==3.8.0
```

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
