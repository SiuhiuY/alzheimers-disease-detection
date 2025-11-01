import os
import numpy as np
import pandas as pd
from .subject_processor import SubjectProcessor


def load_features(label_path, label_map, band="alpha"):
    """Load image-like features and synchronise with class labels.

    Args:
        label_path (str): Path to the .tsv file containing subject labels.
        label_map (dict): Mapping of class names to integer labels.
        band (str): EEG frequency band to load features from.

    Returns:
        tuple: (X_all, y_all, subjects_all)
            where X_all is a numpy array of shape
            (n_total_windows, window_size, height, width)
            [e.g., (n, 3, 32, 32)],
            y_all is a numpy array of shape (n_total_windows,),
            and subjects_all is a numpy array of shape (n_total_windows,).
    """
    # Define data directory based on band
    feature_dir = SubjectProcessor().FEATURES_DIR
    data_dir = os.path.join(feature_dir, band)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Feature directory {data_dir} not found.")

    # Load subject IDs and labels
    label_df = pd.read_csv(label_path, sep="\t")
    label_dict = label_df.set_index("participant_id")['Group'].to_dict()

    # Map class names to integer labels
    label_to_int = {
        subj_id: label_map[group]
        for subj_id, group in label_dict.items()
    }

    # Initialise lists to hold data
    all_features = []
    all_labels = []
    all_subjects = []

    # Load features and match with labels
    feature_files = [
        file for file in os.listdir(data_dir) if file.endswith(".npy")
    ]

    for file_name in feature_files:
        subject_id = file_name.split("_")[0]

        if subject_id in label_to_int:
            # Load features for the subject
            subj_features = np.load(os.path.join(data_dir, file_name))

            # Get the number of windows and label for the subject
            num_windows = subj_features.shape[0]
            label = label_to_int[subject_id]

            # Replicate labels and subject IDs for each window
            subj_labels = [label] * num_windows
            subj_ids = [subject_id] * num_windows

            # Append the subject data to the overall lists
            all_features.extend(subj_features)
            all_labels.extend(subj_labels)
            all_subjects.extend(subj_ids)

    # Convert lists to numpy arrays
    X_all = np.array(all_features, dtype=np.float32)
    y_all = np.array(all_labels, dtype=np.int64)
    subjects_all = np.array(all_subjects, dtype=object)

    return X_all, y_all, subjects_all


if __name__ == "__main__":
    label_path = "data/participants.tsv"
    label_map = {"A": 0, "F": 1, "C": 2}
    features, labels, subjects = load_features(
        label_path, label_map, band="alpha"
    )
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Subjects shape:", subjects.shape)
    print("Unique labels:", np.unique(labels))
    print("Unique subjects:", np.unique(subjects))
    print("Sample features:", features[0].shape)
    print("Sample label:", labels[0])
    print("Sample subject:", subjects[0])
