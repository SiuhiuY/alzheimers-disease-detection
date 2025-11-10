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
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from src.models.EEGNet import EEGNet
from src.eeg_processor import EEGProcessor
from src.subject_processor import SubjectProcessor

# Set random seed
RANDOM_SEED = 123

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)


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
    min_epochs = 0
    max_epochs = 0

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

            if min_epochs == 0 or n_epochs < min_epochs:
                min_epochs = n_epochs
            if n_epochs > max_epochs:
                max_epochs = n_epochs

    # Convert lists to numpy arrays
    X_all = np.concatenate(all_features, axis=0).astype(np.float32)
    y_all = np.array(all_labels, dtype=np.int64)
    subjects_all = np.array(all_subjects, dtype=object)

    print(f"Minimum epochs per subject: {min_epochs}")
    print(f"Maximum epochs per subject: {max_epochs}")

    return X_all, y_all, subjects_all


if __name__ == "__main__":

    # Set seed for reproducibility
    reproducibility(RANDOM_SEED)

    # Define label path and label map
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 0, "C": 1}
    FTD_CN = {"F": 0, "C": 1}
    AD_FTD = {"A": 0, "F": 1}
    label_map = FTD_CN

    # Load EEG data
    X_all, y_all, subjects_all = load_eeg_data(label_map)

    # Define EEGNet input shape parameters
    sampling_rate = 500  # Hz
    epoch_duration = 4   # seconds
    samples = sampling_rate * epoch_duration  # 2000 samples
    chans = X_all.shape[1]  # Number of EEG channels
    kernels = 1  # Single kernel/channel

    # Initialise Leave-One-Subject-Out cross-validator
    logo = LeaveOneGroupOut()
    min_epochs = 0
    all_metrics = []

    for fold, (train_val_ind, test_ind) in enumerate(
        logo.split(X_all, y_all, groups=subjects_all)
    ):
        print(f"Fold {fold + 1}:")

        if len(
            set(np.unique(subjects_all[train_val_ind])) &
            set(np.unique(subjects_all[test_ind]))
        ) != 0:
            raise ValueError("Subjects overlap between train and test sets!")

        X_train_val, X_test = X_all[train_val_ind], X_all[test_ind]
        y_train_val, y_test = y_all[train_val_ind], y_all[test_ind]
        subjects_train_val, subjects_test = (
            subjects_all[train_val_ind], subjects_all[test_ind]
        )

        # Inner train validation split grouped by subjects
        # and stratified by class
        test_size = 0.2
        inner_split = StratifiedGroupKFold(
            n_splits=int(1/test_size), shuffle=True, random_state=RANDOM_SEED
        )

        train_ind, val_ind = next(inner_split.split(
            X_train_val, y_train_val, groups=subjects_train_val
            ))

        if len(
            set(np.unique(subjects_all[train_val_ind][train_ind])) &
            set(np.unique(subjects_all[train_val_ind][val_ind]))
        ) != 0:
            raise ValueError(
                "Subjects overlap between inner train and val sets!"
            )

        X_train, X_val = X_train_val[train_ind], X_train_val[val_ind]
        y_train, y_val = y_train_val[train_ind], y_train_val[val_ind]
        subjects_train, subjects_val = (
            subjects_train_val[train_ind], subjects_train_val[val_ind]
        )

        # Reshape features for EEGNet input
        # EEGNet expects input shape: (n_epochs, n_channels, n_times, 1)
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_val = X_val.reshape(X_val.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        # Convert labels to one-hot encoding
        num_classes = len(label_map)
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # Configure the EEGNet-8,2,16 model with kernel length of 32 samples
        model = EEGNet(nb_classes=num_classes, Chans=chans, Samples=samples,
                       dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                       dropoutType='Dropout')

        # Compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        # Count number of parameters in the model
        numParams = model.count_params()

        # Model checkpoints and early stopping
        # checkpointer = ModelCheckpoint(

        # Model training
        history = model.fit(X_train, y_train, epochs=20,
                            batch_size=64, validation_data=(X_val, y_val),
                            verbose=0)

        # Evaluate the model on the test set
        # model.load_weights('best_model.h5')
        y_pred = model.predict(X_test)

        # Convert predictions and true labels from one-hot to class labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        all_metrics.append((accuracy, precision, recall, f1))

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-score: {f1:.4f}")
        
        
