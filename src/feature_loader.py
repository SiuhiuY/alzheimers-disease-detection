import os
import numpy as np
import pandas as pd


def load_features(label_map, band="alpha"):
    """Load image-like features and synchronise with class labels.

    Args:
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
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
    feature_dir = os.path.join(DATA_DIR, "features")
    data_dir = os.path.join(feature_dir, band)
    label_path = os.path.join(DATA_DIR, "participants.tsv")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Feature directory {data_dir} not found.")

    # Load subject IDs and labels
    label_df = pd.read_csv(label_path, sep="\t")
    label_dict = label_df.set_index("participant_id")["Group"].to_dict()

    # Get valid groups from label_map
    valid_groups = set(label_map.keys())

    # Map class names to integer labels
    label_to_int = {
        subj_id: label_map[group]
        for subj_id, group in label_dict.items() if group in valid_groups
    }

    # Initialise lists to hold data
    all_features = []
    all_labels = []
    all_subjects = []

    # Track number of samples per subject
    min_windows = 0
    max_windows = 0

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
            all_features.append(subj_features)
            all_labels.extend(subj_labels)
            all_subjects.extend(subj_ids)

            if num_windows < min_windows or min_windows == 0:
                min_windows = num_windows
            if num_windows > max_windows:
                max_windows = num_windows

    # Convert lists to numpy arrays
    X_all = np.concatenate(all_features, axis=0).astype(np.float32)
    y_all = np.array(all_labels, dtype=np.int64)
    subjects_all = np.array(all_subjects, dtype=object)

    print(f"Min windows per subject: {min_windows}")
    print(f"Max windows per subject: {max_windows}")

    return X_all, y_all, subjects_all


if __name__ == "__main__":
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 0, "C": 1}
    FTD_CN = {"F": 0, "C": 1}
    features, labels, subjects = load_features(
        label_map=FTD_CN, band="alpha"
    )
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Subjects shape:", subjects.shape)
    print("Unique labels:", np.unique(labels))
    print("Unique subjects:", np.unique(subjects))
    print("Sample features:", features[0].shape)
    print("Sample label:", labels[0])
    print("Sample subject:", subjects[0])
