import os
import mne
import pytest
import numpy as np
from src.eeg_processor import EEGProcessor


@pytest.fixture
def processor():
    """Fixture to create an EEGProcessor instance for testing."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(ROOT_DIR, "..", "data")
    file_path = "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    proc = EEGProcessor(data_folder, file_path)
    proc.load_data()
    proc.epoch_data()
    proc.compute_psd()
    proc.compute_band_psd()
    proc.map_channel_locations()
    proc.interpolate()
    proc.sliding_window()

    # Get number of epochs for later tests
    proc.n_epochs = proc.epochs.get_data().shape[0]

    return proc


def test_load_data(processor):
    """Test loading of EEG data."""
    assert processor.raw is not None

    # Check if raw is an instance of MNE Raw object
    assert isinstance(processor.raw, mne.io.BaseRaw)

    # Check the number of channels
    assert processor.raw.info['nchan'] == 19


def test_load_data_file_not_found():
    """Test loading of EEG data with invalid file path."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(ROOT_DIR, "..", "data")
    invalid_file_path = "non_existent_file.set"
    processor = EEGProcessor(data_folder, invalid_file_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        processor.load_data()


def test_epoch_data(processor):
    """Test epoching of EEG data."""
    assert processor.epochs is not None

    # Check if epochs is an instance of MNE Epochs
    assert isinstance(processor.epochs, mne.Epochs)

    # Check the number of channels
    assert processor.epochs.info['nchan'] == 19

    # Check epoch shape: (n_epochs, n_channels, n_times per epoch)
    assert processor.epochs.get_data().ndim == 3


def test_compute_psd(processor):
    """Test computation of PSD."""
    assert processor.psds is not None
    assert processor.freqs is not None
    assert processor.ch_names is not None

    # Check PSD shape: (n_epochs, n_channels, n_freqs)
    assert processor.psds.ndim == 3
    assert processor.psds.shape[1] == 19

    # Check freqs shape: (n_freqs,)
    assert processor.freqs.ndim == 1
    assert processor.freqs.shape[0] == processor.psds.shape[2]

    # Check frequency range
    assert np.all(processor.freqs >= 0.5) and np.all(processor.freqs <= 40)

    # Check channel names length
    assert len(processor.ch_names) == 19


def test_compute_band_psd(processor):
    """Test computation of band-specific PSD."""
    assert processor.band_psd is not None

    # Check band_psd shape: (n_epochs, n_channels)
    assert processor.band_psd.ndim == 2
    assert processor.band_psd.shape[0] == processor.n_epochs
    assert processor.band_psd.shape[1] == 19


def test_compute_band_psd_invalid_band(processor):
    """Test computation of band-specific PSD with invalid band name."""
    with pytest.raises(ValueError, match="Invalid band name"):
        processor.compute_band_psd(band="invalid_band")


def test_map_channel_locations(processor):
    """Test mapping of channel locations to 2D grid."""
    assert processor.mapped_data is not None

    # Check the shape of the mapped data: (n_epochs, 5, 5)
    assert processor.mapped_data.ndim == 3
    assert processor.mapped_data.shape[0] == processor.n_epochs
    assert processor.mapped_data.shape[1:] == (5, 5)

    # Ensure each grid has nonzero mapped values
    for epoch_grid in processor.mapped_data:
        assert np.any(epoch_grid != 0)


def test_map_channel_locations_invalid(processor):
    """Test mapping of channel locations with invalid channel names."""

    # Temporarily modify channel names to include an invalid name
    original_ch_names = processor.ch_names.copy()
    processor.ch_names[0] = "invalid_channel"

    with pytest.raises(KeyError, match="not in grid mapping"):
        processor.map_channel_locations()

    # Restore original channel names
    processor.ch_names = original_ch_names


def test_interpolate(processor):
    """Test interpolation of mapped data to a finer grid."""
    assert processor.interpolated_data is not None

    # Check the shape of the interpolated data: (n_epochs, 32, 32)
    assert processor.interpolated_data.ndim == 3
    assert processor.interpolated_data.shape[0] == processor.n_epochs
    assert processor.interpolated_data.shape[1:] == (32, 32)

    # Ensure each interpolated grid has nonzero values
    for epoch_interp in processor.interpolated_data:
        assert np.any(epoch_interp != 0)


def test_sliding_windows(processor):
    """Test application of sliding window to interpolated data."""
    assert processor.windows is not None

    # Check the shape of the sliding windows: (n_windows, window_size, 32, 32)
    assert processor.windows.shape[0] > 0
    assert processor.windows.ndim == 4
    assert processor.windows.shape[1] == 3
    assert processor.windows.shape[2:] == (32, 32)


if __name__ == "__main__":
    test_load_data(processor())
    test_epoch_data(processor())
    test_compute_psd(processor())
    test_compute_band_psd(processor())
    test_map_channel_locations(processor())
    test_map_channel_locations_invalid(processor())
    test_interpolate(processor())
    test_sliding_windows(processor())
    print("All tests passed.")
