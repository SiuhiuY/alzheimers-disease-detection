import os
import mne
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from src.models.EEGNet import EEGNet
from src.eeg_processor import EEGProcessor
from src.subject_processor import SubjectProcessor


def reproducibility(seed):
    """Set seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_eeg_data(label_map):
    """Load EEG data for all subjects and prepare for EEGNet.

    Args:
        label_map (dict): Mapping of class names to integer labels.

    Returns:
        tuple: (X_all, y_all, subjects_all)
            where X_all is a numpy array of shape
            (n_total_epochs, n_channels, n_times, 1),
            y_all is a numpy array of shape (n_total_epochs, n_classes),
            and subjects_all is a numpy array of shape (n_total_epochs,).
    """

    # Define directories and labels
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "derivatives")
    LABEL_PATH = os.path.join(DATA_DIR, "participants.tsv")

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data directory {PROCESSED_DIR} not found."
        )

    # Initialise SubjectProcessor to get directories of all subjects
    process_subjects = SubjectProcessor(DATA_DIR)
    process_subjects.find_all_subjects()

    # Load subject IDs and labels
    label_df = pd.read_csv(LABEL_PATH, sep="\t")
    label_dict = label_df.set_index("participant_id")['Group'].to_dict()

    # Get valid groups from label_map
    valid_groups = set(label_map.keys())

    # Map class names to integer labels
    label_to_int = {
            subj_id: label_map[group]
            for subj_id, group in label_dict.items() if group in valid_groups
        }

    # Load EEG data
    all_features = []
    all_labels = []
    all_subjects = []

    # Loop through each subject and process their EEG data
    for subj_dir in process_subjects.subject_dirs:
        subj_id = os.path.basename(subj_dir)

        if subj_id in label_to_int:
            subj_path = os.path.join(PROCESSED_DIR, subj_dir, "eeg",
                                     f"{subj_id}_task-eyesclosed_eeg.set")

            if not os.path.exists(subj_path):
                raise FileNotFoundError(
                    f"EEG file for subject {subj_id} not found at {subj_path}."
                )
            print(f"Loading data for subject {subj_id}...")

            processor = EEGProcessor(PROCESSED_DIR, subj_path)
            processor.load_data()

            # Epoch the data into 4-second segments with 2-second overlap
            processor.epoch_data(duration=4.0, overlap=2.0)
            signals = processor.epochs

            # Extract epoched data as NumPy array
            # Shape: (n_epochs, n_channels, n_times per epoch)
            subj_feature = signals.get_data() * 1000 

            # Get the number of epochs and class label for the subject
            n_epochs = subj_feature.shape[0]
            label = label_to_int[subj_id]

            # Replicate labels and subject IDs for each epoch
            subj_labels = [label] * n_epochs
            subj_ids = [subj_id] * n_epochs

            # Append features, labels, and subject IDs
            all_features.append(subj_feature)
            all_labels.extend(subj_labels)
            all_subjects.extend(subj_ids)

    # Convert lists to numpy arrays
    X_all = np.concatenate(all_features, axis=0).astype(np.float32)
    y_all = np.array(all_labels, dtype=np.int64)
    subjects_all = np.array(all_subjects, dtype=object)

    return X_all, y_all, subjects_all


# Set seed for reproducibility
reproducibility(123)

# Define label path and label map
AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
AD_CN = {"A": 0, "C": 1}
FTD_CN = {"F": 0, "C": 1}

X_all, y_all, subjects_all = load_eeg_data(FTD_CN)

print(X_all.shape, y_all.shape, subjects_all.shape)
print(np.unique(y_all))
print(np.unique(subjects_all))

# Reshape features for EEGNet input
# EEGNet expects input shape: (n_epochs, n_channels, n_times, 1)
# kernels, chans, samples = 1, 19, 151
# X_all = X_all.reshape(X_all.shape[0], chans, samples, kernels)

# # Convert labels to one-hot encoding
# if binary:
#     num_classes = 2
# else:
#     num_classes = 3

# y_all = to_categorical(y_all, num_classes=num_classes)


# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# count number of parameters in the model
numParams    = model.count_params()    

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)


# print(f"Total samples: {X_all.shape[0]}")
# print(f"Feature shape: {X_all.shape[1:]}")  # (n_channels, n_times)
# print(f"Labels shape: {y_all.shape}")
# print(f"Subjects shape: {subjects_all.shape}")
# print(X_all.shape, y_all.shape, subjects_all.shape)
# print("Unique labels:", np.unique(np.argmax(y_all, axis=1)))
# print("Unique subjects:", np.unique(subjects_all))
# print("Sample feature shape:", X_all[0].shape)
# print("Sample label (one-hot):", y_all[0])
