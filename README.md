# Master Thesis - Data Science & Society 

## Project Title
Alzheimer's Disease Detection Using EEG Topographic Images



---

## Project Overview





## Repository Structure
```
THESIS_CODES/
│
├── data/
│   ├── derivatives/
│   ├── features/
│   ├── raw/
│   ├── dataset_description.json
│   ├── participants.json
│   ├── participants.tsv
│   └── CHANGES
│
├── jobs/
│   └── eeg_prep_gpu.sh
│
├── notebooks/
│   ├── Tutorials/
│   ├── eda.ipynb
│   ├── experiment.ipynb
│   ├── image.ipynb
│   └── saved_epo.fif
│
├── src/
│   ├── models/
│   │   ├── AlexNet.py
│   │   ├── CNN.py
│   │   ├── EEGNet.py
│   │   ├── ResNet.py
│   │   └── VGG.py
│   │
│   ├── dataset.py
│   ├── eeg_processor.py
│   ├── feature_loader.py
│   ├── model_trainer.py
│   ├── model_tuner.py
│   ├── subject_processor.py
│   ├── util.py
│   └── __init__.py
│
├── tests/
│   ├── test_subject.py
│   └── __init__.py
│
├── cross_validation.py
├── experiment.py
├── main.py
├── environment.gpu.yml
├── environment.local.yml
├── .gitignore
├── .gitlab-ci.yml
├── README.md
└── requirements.txt

```

## Dataset


## Usage

### Dataset Download 
Download the dataset at http:// 

By choosing your root directory, the dataset will automatically be saved in a "data" folder.

### Image Extraction 
To extract images from the EEG signals: amend the configurations, including band name and the window size, in subject_processor.py and run the command
```
python -m src.subject_processor
```
The images will be saved in a folder which corresponds to the band name of your choice. 

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
│   │   └── delta/
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
│   ├── dataset_description.json
│   ├── participants.json
│   ├── participants.tsv
│   └── CHANGES
```

To test for image extraction steps in eeg_processor.py and subject_processor.py
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
