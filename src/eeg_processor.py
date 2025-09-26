import os
import mne

class EEGProcessor:
    """Class to handle EEG data processing."""
    def __init__(self, data_folder, file_path):
        self.data_folder = data_folder
        self.file_path = file_path
        self.raw = None
        self.epochs = None

    def load_data(self):
        """Load EEG data from a given file path."""
        self.raw = mne.io.read_raw_eeglab(
            os.path.join(
                self.data_folder,
                self.file_path
            ),
            preload=True)
        return self.raw

if __name__ == "__main__":
    data_folder = "E:/EEG"
    file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    processor = EEGProcessor(data_folder, file_path)
    raw_data = processor.load_data()
    print(raw_data)
    print(raw_data.info)
    print(type(raw_data))