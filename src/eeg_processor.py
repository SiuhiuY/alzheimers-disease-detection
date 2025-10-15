import os
import mne
import numpy as np
from scipy.interpolate import griddata


class EEGProcessor:
    """Class to handle EEG data processing.

    Attributes:
        data_folder (str): Path to the folder containing EEG data.
        file_path (str): Path to the EEG data file.
        raw (mne.io.Raw): Raw EEG data.
        epochs (mne.Epochs): Epoched EEG data.
        sfreq (float): Sampling frequency of the EEG data.
    """

    def __init__(self, data_folder, file_path):
        """
        Initialize the EEGProcessor with data folder and file path.

        Args:
            data_folder (str): Path to the folder containing EEG data.
            file_path (str): Path to the EEG data file.
        """
        self.data_folder = data_folder
        self.file_path = file_path
        self.raw = None
        self.epochs = None

    def load_data(self):
        """Load EEG data from a given file path.

        Returns:
            mne.io.Raw: Loaded raw EEG data."""
        self.raw = mne.io.read_raw_eeglab(
            os.path.join(
                self.data_folder,
                self.file_path
            ),
            preload=True)
        return self.raw

    def epoch_data(self, duration=1.0, overlap=0.0):
        """Epoch the raw data by splitting it into 1-second segments.

        Args:
            duration (float): Duration of each epoch in seconds.
            overlap (float): Overlap between consecutive epochs in seconds.
        Returns:
            mne.Epochs: Epoched EEG data.
        """
        self.epochs = mne.make_fixed_length_epochs(
            self.raw,
            duration=duration,
            overlap=overlap,
            preload=True)
        return self.epochs

    def compute_psd(self, fmin=0.5, fmax=40):
        """Compute the Power Spectral Density (PSD) of the epochs.

        Args:
            band (str): Frequency band of interest.
            fmin (float): Minimum frequency of interest.
            fmax (float): Maximum frequency of interest.
        Returns:
            tuple: (psds, freqs) where psds is the PSD values and
                freqs are the corresponding frequencies.
        """
        psd = self.epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax)
        psds, freqs = psd.get_data(return_freqs=True)
        return psds, freqs

    def compute_band_power(self, band="alpha", psds=None, freqs=None):
        """Compute the average power in standard EEG frequency bands.

        Args:
            band (str): Frequency band of interest.
            psds (np.ndarray): Power Spectral Density values.
            freqs (np.ndarray): Corresponding frequencies.
        Returns:
            np.ndarray: Average power in the specified frequency band.
        """
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        fmin, fmax = bands.get(band)
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, :, mask].mean(axis=-1)
        return band_power


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(ROOT_DIR, "..", "data")
    file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    # data_folder = os.path.join(os.path.expanduser("~"), "EEG")
    # file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    # if not os.path.exists(os.path.join(data_folder, file_path)):
    #     data_folder = "E:/EEG"
    processor = EEGProcessor(data_folder, file_path)
    raw_data = processor.load_data()
    epochs_data = processor.epoch_data()
    psds, freqs = processor.compute_psd()
    band_power = processor.compute_band_power(band="alpha", psds=psds, freqs=freqs)
    print(psds.shape)
    print(freqs)
    print(band_power.shape)
