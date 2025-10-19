import os
import mne
import numpy as np
from scipy.interpolate import RectBivariateSpline


class EEGProcessor:
    """Class to transform EEG signals into image-like topomaps."""

    def __init__(self, data_folder, file_path):
        """Initialize the EEGProcessor with data folder and file path.

        Args:
            data_folder (str): Path to the folder containing EEG data.
            file_path (str): Path to the file of a specific subject.

        Raises:
            ValueError: If data_folder or file_path is not provided.
        """
        if data_folder is None or file_path is None:
            raise ValueError("data_folder and file_path must be provided.")
        self.data_folder = data_folder
        self.file_path = file_path

    def __repr__(self):
        """String representation of the EEGProcessor.
        """
        return (f"EEGProcessor(data_folder={self.data_folder}, "
                f"file_path={self.file_path})")

    def load_data(self):
        """Load EEG data from a given file path.

        Returns:
            mne.io.Raw: Loaded EEG signals.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        full_path = os.path.join(self.data_folder, self.file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File {full_path} does not exist.")
        self.raw = mne.io.read_raw_eeglab(full_path, preload=True)
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

    def compute_psd(self):
        """Compute the Power Spectral Density (PSD) of the epochs.

        Returns:
            tuple: (psds, freqs, ch_names) where psds is the PSD values,
                freqs are the corresponding frequencies
                and ch_names are the channel names.
        """
        self.psd = self.epochs.compute_psd(
            method="welch",
            fmin=0.5,
            fmax=40)
        self.ch_names = self.psd.info['ch_names']  # Get channel names

        # Convert PSD to numpy arrays with .get_data()
        # psds.shape: (n_epochs, n_channels, n_freqs)
        # freqs.shape: (n_freqs, )
        self.psds, self.freqs = self.psd.get_data(return_freqs=True)
        return self.psds, self.freqs, self.ch_names

    def compute_band_psd(self, band="alpha"):
        """Compute the average power in standard EEG frequency bands.

        Args:
            band (str): Frequency band of interest.

        Returns:
            np.ndarray: PSD values in the specified frequency band.

        Raises:
            ValueError: If an invalid band name is provided.
        """
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }

        if band not in bands:
            raise ValueError(
                f"Invalid band name. "
                f"Choose from {list(bands.keys())}."
            )

        # Get frequency band mask
        fmin, fmax = bands.get(band)
        mask = (self.freqs >= fmin) & (self.freqs <= fmax)

        # Average PSD across the selected frequency bins
        # (collapse last dimension)
        # band_psd.shape: (n_epochs, n_channels)
        self.band_psd = self.psds[:, :, mask].mean(axis=-1)
        return self.band_psd

    def map_channel_locations(self):
        """Map EEG channel locations to a 5x5 grid.

        Returns:
            np.ndarray: 5x5 grids for each epoch with mapped PSD values.

        Raises:
            KeyError: If a channel name is not found in the grid mapping.
        """
        n_epochs = self.band_psd.shape[0]
        self.mapped_data = np.zeros((n_epochs, 5, 5))

        # Create a 5x5 grid for standard 10-20 system channels
        grid_mapping = {
            "Fp1": (0, 1), "Fp2": (0, 3),
            "F7": (1, 0), "F3": (1, 1), "Fz": (1, 2), "F4": (1, 3), "F8": (1, 4),
            "T3": (2, 0), "C3": (2, 1), "Cz": (2, 2), "C4": (2, 3), "T4": (2, 4),
            "T5": (3, 0), "P3": (3, 1), "Pz": (3, 2), "P4": (3, 3), "T6": (3, 4),
            "O1": (4, 1), "O2": (4, 3)
        }

        # Map the PSD values onto the grid
        for epoch_ind, epoch in enumerate(self.band_psd):
            grid = np.zeros((5, 5))
            # Match channel names to grid positions
            for ch_index, ch_name in enumerate(self.ch_names):
                if ch_name not in grid_mapping:
                    raise KeyError(f"Channel {ch_name} not in grid mapping.")
                x, y = grid_mapping.get(ch_name)
                grid[x, y] = epoch[ch_index]
            self.mapped_data[epoch_ind] = grid

        # output shape: (n_epochs, 5, 5)
        return self.mapped_data

    def interpolate(self, grid_size=(32, 32)):
        """Interpolate the 2D grid data for spatial smoothing.

        Args:
            grid_size (tuple): Size of the output grid (rows, cols).

        Returns:
            np.ndarray: Interpolated data
        """
        n_epochs, rows, cols = self.mapped_data.shape

        # Original grid coordinates
        x = np.linspace(0, rows-1, rows)
        y = np.linspace(0, cols-1, cols)

        # New grid coordinates
        xi = np.linspace(0, rows-1, grid_size[0])
        yi = np.linspace(0, cols-1, grid_size[1])

        # Initialise empty array for interpolated data
        self.interpolated_data = np.zeros(
            (n_epochs, grid_size[0], grid_size[1])
        )

        for epoch_ind in range(n_epochs):
            # Create the spline interpolation function
            interp = RectBivariateSpline(
                x, y,
                self.mapped_data[epoch_ind]
            )
            # Interpolate to new grid
            self.interpolated_data[epoch_ind] = interp(xi, yi)

        # output shape: (n_epochs, 32, 32)
        return self.interpolated_data

    def sliding_window(self, window_size=3, step_size=1):
        """Apply sliding window to the data.

        Args:
            window_size (int): Size of the sliding window.
            step_size (int): Step size for the sliding window.

        Returns:
            np.ndarray: Data after applying sliding window.
        """
        n_epochs = self.interpolated_data.shape[0]
        windows = []
        for start in range(0, n_epochs - window_size + 1, step_size):
            end = start + window_size
            # shape: (window_size, 32, 32)
            window = self.interpolated_data[start:end]
            windows.append(window)

        # output shape: (n_windows, window_size, 32, 32)
        self.windows = np.array(windows)
        return self.windows


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(ROOT_DIR, "..", "data")
    file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    processor = EEGProcessor(data_folder, file_path)
    processor.load_data()
    processor.epoch_data()
    processor.compute_psd()
    processor.compute_band_psd()
    processor.map_channel_locations()
    processor.interpolate()
    sliding_windows = processor.sliding_window()
    print(processor)
    print("Sliding windows shape:", sliding_windows.shape)
