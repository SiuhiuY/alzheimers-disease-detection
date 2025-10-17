import glob
import os
import numpy as np
from .eeg_processor import EEGProcessor


class SubjectProcessor:
    """Class to process EEG data for all subjects.
    """

    def __init__(self, data_root=None):
        """
        Initialize the SubjectProcessor with data folder.

        Args:
            data_root (str): Path to the root data folder.
        """
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "..", "data")
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, "derivatives")
        self.FEATURES_DIR = os.path.join(self.DATA_DIR, "features")
        os.makedirs(self.FEATURES_DIR, exist_ok=True)

        self.subject_files = []
        self.band_name = None

    def find_all_subjects(self):
        """
        Find all subject files in the data folder.

        Returns:
            list: List of subject file paths.
        """
        self.subject_files = glob.glob(
            os.path.join(self.PROCESSED_DIR, "sub-*/eeg/*.set")
        )
        return self.subject_files


if __name__ == "__main__":
    processor = SubjectProcessor()
    subjects = processor.find_all_subjects()
    print("Found subjects:", len(subjects))
