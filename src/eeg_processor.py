import os
import mne
import numpy as np
from scipy.interpolate import RectBivariateSpline


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
            fmin (float): Minimum frequency of interest.
            fmax (float): Maximum frequency of interest.
        Returns:
            tuple: (psds, freqs, ch_names) where psds is the PSD values and
                freqs are the corresponding frequencies
                and ch_names are the channel names.
        """
        psd = self.epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax)
        ch_names = psd.info['ch_names']  # Get channel names

        # psds.shape: (n_epochs, n_channels, n_freqs)
        psds, freqs = psd.get_data(return_freqs=True)
        return psds, freqs, ch_names

    def compute_band_psd(self, band="alpha", psds=None, freqs=None):
        """Compute the average power in standard EEG frequency bands.

        Args:
            band (str): Frequency band of interest.
            psds (np.ndarray): Power Spectral Density values.
            freqs (np.ndarray): Corresponding frequencies.
        Returns:
            np.ndarray: PSD values in the specified frequency band.
        """
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }

        # Get frequency band mask
        fmin, fmax = bands.get(band)
        mask = (freqs >= fmin) & (freqs <= fmax)

        # band_psd.shape: (n_epochs, n_channels)
        band_psd = psds[:, :, mask].mean(axis=-1)
        return band_psd

    def map_channel_locations(self, band_psd, ch_names):
        """Map EEG channel locations to 2D coordinates.

        Args:
            band_psd (np.ndarray): Band power values with shape (n_epochs, n_channels).
            ch_names (list): List of channel names.
        Returns:
            np.ndarray: 2D coordinates of EEG channels.
        """
        n_epochs = band_psd.shape[0]
        mapped_data = np.zeros((n_epochs, 5, 5))

        # Create a 5x5 grid for standard 10-20 system channels
        grid_mapping = {
            "Fp1": (0, 1), "Fp2": (0, 3),
            "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
            "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
            "T5": (3, 0), "P3": (3, 1), "Pz": (3, 2), "P4": (3, 3), "T6": (3, 4),
            "O1": (4, 1), "O2": (4, 3)
        }

        # Map the PSD values onto the grid
        for epoch_ind, epoch in enumerate(band_psd):
            grid = np.zeros((5, 5))
            for ch_index, ch_name in enumerate(ch_names):
                x, y = grid_mapping.get(ch_name)
                grid[x, y] = epoch[ch_index]
            mapped_data[epoch_ind] = grid
        return mapped_data

    def interpolate(self, mapped_data, grid_size=(32, 32)):
        """Interpolate the 2D grid data for spatial smoothing.

        Args:
            mapped_data (np.ndarray): 2D grid data with shape (n_epochs, rows, cols).
            grid_size (tuple): Size of the output grid (rows, cols).
        Returns:
            np.ndarray: Interpolated data on a 2D grid.
        """
        n_epochs, rows, cols = mapped_data.shape

        # Original grid coordinates
        x = np.linspace(0, rows-1, rows)
        y = np.linspace(0, cols-1, cols)

        # New grid coordinates
        xi = np.linspace(0, rows-1, grid_size[0])
        yi = np.linspace(0, cols-1, grid_size[1])

        # Initialise empty array for interpolated data
        interpolated_data = np.zeros((n_epochs, grid_size[0], grid_size[1]))

        for epoch_ind in range(n_epochs):
            # Create the spline interpolation function
            interp = RectBivariateSpline(
                y, x,
                mapped_data[epoch_ind]
            )
            # Interpolate to new grid
            interpolated_data[epoch_ind] = interp(yi, xi)
        return interpolated_data



if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(ROOT_DIR, "..", "data")
    file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    processor = EEGProcessor(data_folder, file_path)
    raw_data = processor.load_data()
    epochs_data = processor.epoch_data()
    psds, freqs, ch_names = processor.compute_psd()
    band_psd = processor.compute_band_psd(band="alpha", psds=psds, freqs=freqs)
    grid = processor.map_channel_locations(band_psd, ch_names)
    interpolated_grid = processor.interpolate(grid, grid_size=(32, 32))
    print(interpolated_grid.shape)
    print(interpolated_grid[:3])

