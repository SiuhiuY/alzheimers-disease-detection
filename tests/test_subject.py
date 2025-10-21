import os
import numpy as np
import pytest
from src.subject_processor import SubjectProcessor


@pytest.fixture
def sub_processor():
    """Fixture to create a SubjectProcessor instance for testing."""
    sub_proc = SubjectProcessor()
    return sub_proc


def test_find_all_subjects(sub_processor):
    """Test finding all subject directories."""
    subjects = sub_processor.find_all_subjects()
    assert subjects is not None
    assert len(subjects) > 0
    for subj in subjects:
        full_path = os.path.join(sub_processor.PROCESSED_DIR, subj)

        # Check if the subject directory exists
        assert os.path.isdir(full_path)

        # Check if the subject directory name starts with 'sub-'
        assert "sub-" in os.path.basename(subj)


def test_find_all_subjects_no_dir(sub_processor):
    """Test finding subjects when the processed directory does not exist."""

    # Temporarily change the PROCESSED_DIR to a non-existent path
    original_dir = sub_processor.PROCESSED_DIR
    sub_processor.PROCESSED_DIR = "non_existent_directory"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        sub_processor.find_all_subjects()

    # Restore the original directory
    sub_processor.PROCESSED_DIR = original_dir


def test_choose_band(sub_processor):
    """Test choosing a valid frequency band."""
    # Test default band selection
    sub_processor.choose_band()
    assert sub_processor.band_name == "alpha"

    # Test specific band selection
    sub_processor.choose_band(band_name="theta")
    assert sub_processor.band_name == "theta"


def test_choose_band_invalid(sub_processor):
    """Test choosing an invalid frequency band."""
    with pytest.raises(ValueError, match="Invalid band name"):
        sub_processor.choose_band(band_name="invalid_band")


def test_choose_window_size(sub_processor):
    """Test setting window and step size."""
    sub_processor.choose_window_size(5, 2)
    assert sub_processor.window_size == 5
    assert sub_processor.step_size == 2


def test_choose_window_size_invalid(sub_processor):
    """Test choosing an invalid window or step size."""
    with pytest.raises(TypeError, match="must be integers"):
        sub_processor.choose_window_size(3.5, 1)
    with pytest.raises(TypeError, match="must be integers"):
        sub_processor.choose_window_size("3", 1)
    with pytest.raises(ValueError, match="must be positive integers"):
        sub_processor.choose_window_size(-3, 1)
    with pytest.raises(
        ValueError,
        match="Step size must be less than or equal to window size"
    ):
        sub_processor.choose_window_size(2, 3)


def test_prepare_output_folder(sub_processor, tmp_path):
    """Test preparation of output folder."""
    # Override FEATURES_DIR to a temporary path for testing
    sub_processor.FEATURES_DIR = os.path.join(tmp_path, "features")
    sub_processor.choose_band("beta")
    band_folder = sub_processor.prepare_output_folder()
    assert os.path.exists(band_folder)
    assert os.path.isdir(band_folder)

    # Check if it is a clean folder
    assert len(os.listdir(band_folder)) == 0

    # Check correct path
    assert band_folder == os.path.join(
        sub_processor.FEATURES_DIR, sub_processor.band_name
    )


def test_save_data(sub_processor, tmp_path):
    """Test saving processed data."""
    # Create dummy data to save
    dummy_data = np.random.rand(10, 32, 32)
    save_path = os.path.join(tmp_path, "test_data.npy")
    sub_processor.save_data(dummy_data, save_path)

    # Check if the file was created
    assert os.path.exists(save_path)

    # Load the saved data and verify its content
    loaded_data = np.load(save_path)
    assert np.array_equal(loaded_data, dummy_data)


def test_process_single_subject(sub_processor, tmp_path):
    """Test processing a single subject."""
    # Override FEATURES_DIR to a temporary path for testing
    sub_processor.FEATURES_DIR = os.path.join(tmp_path, "features")
    sub_processor.find_all_subjects()
    sub_processor.choose_band("theta")
    sub_processor.choose_window_size()
    sub_processor.prepare_output_folder()

    # Use the first subject directory for testing
    subject_dir = sub_processor.subject_dirs[0]
    subject_id = os.path.basename(subject_dir)
    subject_path = os.path.join(
        subject_dir, "eeg",
        f"{subject_id}_task-eyesclosed_eeg.set"
    )
    assert os.path.exists(subject_path)

    # Run processing on the subject
    sub_processor.process_single_subject(subject_dir)

    # Check if the processed file was created
    save_path = os.path.join(
        sub_processor.FEATURES_DIR, sub_processor.band_name,
        f"{subject_id}_{sub_processor.band_name}_psd.npy"
    )
    assert os.path.exists(save_path)

    # Verify the content of the saved file
    # sliding windows shape: (n_windows, window_size, 32, 32)
    loaded_data = np.load(save_path)
    assert loaded_data.ndim == 4
    assert loaded_data.shape[1] == sub_processor.window_size
    assert loaded_data.shape[2:] == (32, 32)


def test_process_all_subjects(sub_processor, tmp_path):
    """Test processing all subjects."""
    # Override FEATURES_DIR to a temporary path for testing
    sub_processor.FEATURES_DIR = os.path.join(tmp_path, "features")
    sub_processor.find_all_subjects()
    sub_processor.choose_band("beta")
    sub_processor.choose_window_size()
    sub_processor.prepare_output_folder()

    # Run the processing for the test subjects
    sub_processor.process_all_subjects()

    # Check if processed files were created for each subject
    for subject_dir in sub_processor.subject_dirs:
        subject_id = os.path.basename(subject_dir)
        save_path = os.path.join(
            sub_processor.FEATURES_DIR, sub_processor.band_name,
            f"{subject_id}_{sub_processor.band_name}_psd.npy"
        )
        assert os.path.exists(save_path)
