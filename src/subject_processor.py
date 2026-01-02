import os
import numpy as np
from .eeg_processor import EEGProcessor


class SubjectProcessor:
    """Class to process EEG data for all subjects.

    This class handles the processing of EEG data for multiple subjects,
    including loading data, selecting frequency bands,
    configuring sliding windows, and saving the processed data.

     Attributes:
        ROOT_DIR (str): Directory of the current script file.
        DATA_DIR (str): Data directory containing subject folders.
        PROCESSED_DIR (str): Directory for preprocessed EEG data.
        FEATURES_DIR (str): Directory for extracted features.
        subject_dirs (list): List of subject directory paths.
        band_name (str): Frequency band of interest."""

    def __init__(self, data_dir, feature_type="psd"):
        """Initialize the SubjectProcessor.
        """
        self.DATA_DIR = data_dir
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, "derivatives")
        self.FEATURES_DIR = os.path.join(self.DATA_DIR, "features")

        # Create directory for storing features
        os.makedirs(self.FEATURES_DIR, exist_ok=True)

        self.subject_dirs = []
        self.band_name = None
        self.feature_type = feature_type

    def __repr__(self):
        """String representation of the SubjectProcessor.
        """
        return (f"SubjectProcessor(features_dir={self.FEATURES_DIR})")

    def find_all_subjects(self):
        """Find all subject directories in the data folder.

        Returns:
            list: List of subject directory paths.

        Raises:
            FileNotFoundError: If the processed directory does not exist.
        """
        if not os.path.exists(self.PROCESSED_DIR):
            raise FileNotFoundError(
                f"Processed directory {self.PROCESSED_DIR} does not exist."
            )

        # List directories for all subjects
        self.subject_dirs = os.listdir(self.PROCESSED_DIR)
        return self.subject_dirs

    def choose_band(self, band_name="alpha"):
        """Choose the frequency band of interest.

        Args:
            band_name (str): Name of the frequency band.

        Raises:
            ValueError: If the band name is invalid.
        """
        if band_name.lower() not in EEGProcessor.VALID_BANDS:
            valid_bands = list(EEGProcessor.VALID_BANDS.keys())
            raise ValueError(
                f"Invalid band name. Choose from {valid_bands}."
            )

        self.band_name = band_name

    def choose_window_size(self, window_size=4, step_size=1):
        """Choose sliding window parameters.

        Args:
            window_size (int): Size of the sliding window in seconds.
            step_size (int): Step size for the sliding window in seconds.
        """
        if not isinstance(window_size, int) or not isinstance(step_size, int):
            raise TypeError("Window size and step size must be integers.")
        if window_size <= 0 or step_size <= 0:
            raise ValueError(
                "Window size and step size must be positive integers."
            )
        if step_size > window_size:
            raise ValueError(
                "Step size must be less than or equal to window size."
            )

        self.window_size = window_size
        self.step_size = step_size

    def prepare_output_folder(self):
        """Create output directory for the selected band.

        Returns:
            band_folder (str): Path to the output directory for the
                selected band.
        """
        self.band_folder = os.path.join(self.FEATURES_DIR, self.band_name)
        os.makedirs(self.band_folder, exist_ok=True)
        return self.band_folder

    def save_data(self, data, save_path):
        """Save processed subject data to a file.

        Args:
            data (np.ndarray): Processed data to save.
            save_path (str): Path to save the data.
        """
        with open(save_path, "wb") as f:
            np.save(f, data)

    def process_single_subject(self, subject_dir):
        """Process a single subject and save the processed data.
        Args:
            subject_file (str): Path to the subject EEG file.
        """
        # Get the subject file path
        self.subject_id = os.path.basename(subject_dir)
        self.subject_path = os.path.join(
            subject_dir, "eeg",
            f"{self.subject_id}_task-eyesclosed_eeg.set"
        )

        # Call EEGProcessor for the subject
        processor = EEGProcessor(self.PROCESSED_DIR, self.subject_path)
        processor.load_data()
        processor.epoch_data()
        processor.compute_psd()
        if self.feature_type == "psd":
            processor.compute_band_psd(band=self.band_name)
        if self.feature_type == "relative":
            processor.compute_relative_band_power(band=self.band_name)
        processor.map_channel_locations()
        processor.interpolate()
        sliding_windows = processor.sliding_window(
            window_size=self.window_size,
            step_size=self.step_size
        )

        # Save the transformed data
        save_path = os.path.join(
            self.FEATURES_DIR, self.band_name,
            f"{self.subject_id}_{self.band_name}_psd.npy"
        )
        self.save_data(sliding_windows, save_path)

    def process_all_subjects(self):
        """Process all subjects and save their data.
        """
        for subject_dir in self.subject_dirs:
            self.process_single_subject(subject_dir)


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(ROOT_DIR, "..", "data")
    processor = SubjectProcessor(data_dir, feature_type="psd")
    processor.find_all_subjects()
    print("Found subjects:", len(processor.subject_dirs))
    processor.choose_band(band_name="alpha")
    print("Chosen band:", processor.band_name)
    processor.choose_window_size(window_size=4)
    print("Window size set to:", processor.window_size)
    processor.prepare_output_folder()
    processor.process_all_subjects()
    print("Processing completed for all subjects.")
