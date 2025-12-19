import os
import mne
import numpy as np
import pandas as pd
import random
import wandb
import optuna
import logging
import gc
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping)
from tensorflow.keras import backend as K
from wandb.integration.keras import WandbMetricsLogger
from optuna.integration import KerasPruningCallback
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils import class_weight
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from src.models.EEGNet import EEGNet
from src.eeg_processor import EEGProcessor
from src.subject_processor import SubjectProcessor

# Reference:
# https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna
# https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893

# ----------------------------------------------------------------------------
#                          Configuration Settings
# ----------------------------------------------------------------------------

RANDOM_SEED = 123

# GPU configuration
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# Remove unwanted warnings
logging.getLogger("tensorflow").disabled = True


# ----------------------------------------------------------------------------
#                             Helper Functions
# ----------------------------------------------------------------------------

def reproducibility(seed):
    """Set seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value to set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_eeg_data(label_map, chans, samples, kernels, downsample_to=None):
    """Load EEG data for all subjects and prepare for EEGNet.

    Args:
        label_map (dict): Mapping of class names to integer labels.
        chans (int): Number of EEG channels.
        samples (int): Number of time steps per epoch.
        kernels (int): Number of kernels for EEGNet.
        downsample_to (int, optional): Frequency to downsample the data to.
            If None, no downsampling is performed. Defaults to None.

    Returns:
        tuple: (X_all, y_all, subjects_all)
            where X_all is a numpy array of shape
            (n_total_epochs, n_channels, n_times, 1),
            y_all is a numpy array of shape (n_total_epochs,),
            and subjects_all is a numpy array of shape (n_total_epochs,).
    """

    # Define directories and labels
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "derivatives")
    LABEL_PATH = os.path.join(DATA_DIR, "participants.tsv")

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data directory {PROCESSED_DIR} not found."
        )

    # Initialise SubjectProcessor to get directories of all subjects
    process_subjects = SubjectProcessor(DATA_DIR)
    process_subjects.find_all_subjects()

    # Load subject IDs and labels
    label_df = pd.read_csv(LABEL_PATH, sep="\t")
    label_dict = label_df.set_index("participant_id")["Group"].to_dict()

    # Get valid groups from label_map
    valid_groups = set(label_map.keys())

    # Map class names to integer labels
    label_to_int = {
            subj_id: label_map[group]
            for subj_id, group in label_dict.items() if group in valid_groups
        }

    # Load EEG data
    all_features = []
    all_labels = []
    all_subjects = []
    min_epochs = 0
    max_epochs = 0

    # Loop through each subject and process their EEG data
    for subj_dir in process_subjects.subject_dirs:
        subj_id = os.path.basename(subj_dir)

        if subj_id in label_to_int:
            subj_path = os.path.join(subj_dir, "eeg",
                                     f"{subj_id}_task-eyesclosed_eeg.set")
            processor = EEGProcessor(PROCESSED_DIR, subj_path)
            processor.load_data()
            raw = processor.raw

            if downsample_to is not None:
                raw.resample(downsample_to)

            # Epoch the data into 4-second segments with 2-second overlap
            processor.epoch_data(duration=4.0, overlap=2.0)
            signals = processor.epochs

            # Extract epoched data as NumPy array
            # Shape: (n_epochs, n_channels, n_times per epoch)
            # Scale the signal values by 1000 as in the EEGNet source code
            subj_feature = signals.get_data() * 1000

            # Get the number of epochs and class label for the subject
            n_epochs = subj_feature.shape[0]
            label = label_to_int[subj_id]

            # Replicate labels and subject IDs for each epoch
            subj_labels = [label] * n_epochs
            subj_ids = [subj_id] * n_epochs

            # Append features, labels, and subject IDs
            all_features.append(subj_feature)
            all_labels.extend(subj_labels)
            all_subjects.extend(subj_ids)

            if min_epochs == 0 or n_epochs < min_epochs:
                min_epochs = n_epochs
            if n_epochs > max_epochs:
                max_epochs = n_epochs

    # Convert lists to numpy arrays
    X_all = np.concatenate(all_features, axis=0).astype(np.float32)
    y_all = np.array(all_labels, dtype=np.int64)
    subjects_all = np.array(all_subjects, dtype=object)

    # Reshape X_all for EEGNet input: (n_epochs, n_channels, n_times, 1)
    X_all = X_all.reshape(X_all.shape[0], chans, samples, kernels)

    print(f"Minimum epochs per subject: {min_epochs}")
    print(f"Maximum epochs per subject: {max_epochs}")

    return X_all, y_all, subjects_all


def calculate_metrics(model, X_test, y_test, threshold=0.5):
    """Calculate evaluation metrics on the test set.

    Args:
        model (tf.keras.Model): Trained EEGNet model
        X_test (np.ndarray): Test features of shape
                             (n_samples, n_channels, n_times, 1)
        y_test (np.ndarray): True labels of shape (n_samples,)
        threshold (float): Threshold for binary classification

    Returns:
        dict: Dictionary containing accuracy, precision, recall,
              f1_score, auc, and confusion_matrix.
    """
    # Get model predictions
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_true": y_test,
        "y_pred_probs": y_pred_probs,
        "y_pred": y_pred
    }


def calculate_class_weights(y):
    """Compute class weights.
    Args:
        y (np.ndarray): Array of shape (n_samples,) containing class labels.

    Returns:
        dict: Dictionary mapping class indices to weights.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weight_dict = dict(zip(classes, weights))

    return class_weight_dict


def get_optimizer(optimizer_name, lr):
    """Get the optimizer instance based on the name.

    Args:
        optimizer_name (str): Name of the optimizer
            ("Adam", "RMSprop", "AdamW").
        lr (float): Learning rate.

    Returns:
        tf.keras.optimizers.Optimizer: Optimizer instance.
    """
    optimiser_dict = {
        "adam": Adam(learning_rate=lr),
        "rmsprop": RMSprop(learning_rate=lr),
        "adamw": AdamW(learning_rate=lr)
    }
    return optimiser_dict.get(optimizer_name.lower())


def build_model(params, num_classes, chans, samples):
    """Build and compile the EEGNet model.

    Args:
        params (dict): Hyperparameters for the model.
        num_classes (int): Number of output classes.
        chans (int): Number of EEG channels.
        samples (int): Number of time samples per epoch.

    Returns:
        Model: Compiled Keras EEGNet model.
    """
    # Suggest hyperparameters
    F1 = params["F1"]
    D = params["D"]
    F2 = F1 * D
    dropoutRate = params["dropoutRate"]
    kernLength = params["kernLength"]
    dropoutType = params["dropoutType"]
    lr = params["learning_rate"]
    optimizer_name = params["optimizer"]
    optimizer = get_optimizer(optimizer_name, lr)

    # Build EEGNet model
    model = EEGNet(nb_classes=num_classes,
                   Chans=chans,
                   Samples=samples,
                   dropoutRate=dropoutRate,
                   kernLength=kernLength,
                   F1=F1, D=D, F2=F2,
                   dropoutType=dropoutType)

    # Compile the model
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model


# ----------------------------------------------------------------------------
#                       Optuna Objective Function
# ----------------------------------------------------------------------------

def objective(trial, outer_fold, X_train_val, y_train_val,
              subjects_train_val, num_classes, chans, samples, class_names
              ):
    """

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
        outer_fold (int): Current outer fold index.
        inner_cv (StratifiedGroupKFold): Inner cross-validation splitter.
        X_train_val (np.ndarray): Training and validation features.
        y_train_val (np.ndarray): Training and validation labels.
        subjects_train_val (np.ndarray):
            Subject IDs for training/validation data.
        num_classes (int): Number of output classes.
        chans (int): Number of EEG channels.
        samples (int): Number of time samples per epoch.
        class_names (list): List of class names.

    Returns:
        float: Validation accuracy for the trial.
    """
    # Clear clutter from previous session graphs
    K.clear_session()

    # Define callbacks
    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=10,
                                   restore_best_weights=True,
                                   start_from_epoch=3,
                                   verbose=0)

    # Suggest hyperparameters
    params = {
        "F1": trial.suggest_int("F1", 5, 15),
        "D": trial.suggest_int("D", 1, 4),
        "dropoutRate": trial.suggest_categorical("dropoutRate", [0.25]),
        "kernLength": trial.suggest_categorical("kernLength", [64]),
        "dropoutType": trial.suggest_categorical(
            "dropoutType", ["Dropout"]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 3e-5, 5e-4, log=True),
        "optimizer": trial.suggest_categorical(
            "optimizer", ["adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_categorical("epochs", [10, 20, 30])
        }

    # Initialise wandb logging for monitoring learning process
    config = params
    config["trial.number"] = trial.number
    wandb.init(
        project=f"EEGNet_Nested_CV_{'_'.join(class_names)}_v2",
        name=f"Outer_Fold{outer_fold + 1}_Trial_{trial.number + 1}",
        config=config,
        group=f"Outer_Fold_{outer_fold + 1}",
        job_type="tuning",
        reinit=True
    )

    # Single train/val split
    train_idx, val_idx = next(single_split.split(
        X_train_val, y_train_val, groups=subjects_train_val
    ))

    if len(set(np.unique(subjects_train_val[train_idx])) &
           set(np.unique(subjects_train_val[val_idx]))) != 0:
        raise ValueError(
            "Subjects overlap between train and validation sets!"
        )

    # Build training and validation sets
    X_train, y_train, _ = (
        X_train_val[train_idx],
        y_train_val[train_idx],
        subjects_train_val[train_idx]
    )

    X_val, y_val, _ = (
        X_train_val[val_idx],
        y_train_val[val_idx],
        subjects_train_val[val_idx]
    )

    # Build model
    model = build_model(params, num_classes, chans, samples)

    # Train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=params["batch_size"],
                        epochs=params["epochs"],
                        # class_weight=calculate_class_weights(y_train),
                        callbacks=[
                            KerasPruningCallback(trial, "val_loss"),
                            early_stopping],
                        verbose=0)

    # Log training history to WandB
    for epoch in range(len(history.history["loss"])):
        wandb.log({
            "train_loss": history.history["loss"][epoch],
            "train_accuracy": history.history["accuracy"][epoch],
            "val_loss": history.history["val_loss"][epoch],
            "val_accuracy": history.history["val_accuracy"][epoch]
        }, step=epoch)

    if trial.should_prune():
        wandb.run.summary["state"] = "pruned"
        wandb.finish(quiet=True)
        raise optuna.TrialPruned()

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Trial {trial.number} - Validation Accuracy: {val_acc}")

    wandb.run.summary["val_accuracy"] = val_acc
    wandb.run.summary["state"] = "completed"
    wandb.finish()

    # Clear session
    K.clear_session()
    gc.collect()

    return val_acc


# ----------------------------------------------------------------------------
#                               Main Execution
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # Set seeds for reproducibility
    reproducibility(RANDOM_SEED)

    # Define labels and load data
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 1, "C": 0}
    FTD_CN = {"F": 1, "C": 0}
    AD_FTD = {"A": 1, "F": 0}
    label_map = AD_CN
    model_name = f"EEGNet_{'_'.join(label_map.keys())}"
    num_classes = 1  # Binary classification
    class_names = [
        k for k, v in sorted(
            label_map.items(), key=lambda item: item[1]
        )
    ]

    # Define EEGNet input shape parameters
    sampling_rate = 128  # Hz
    epoch_duration = 4   # seconds
    samples = sampling_rate * epoch_duration  # 1000 samples
    chans = 19  # Number of EEG channels
    kernels = 1  # Single kernel/channel

    print("Loading EEG data...")
    X_all, y_all, subjects_all = load_eeg_data(label_map, chans, samples,
                                               kernels,
                                               downsample_to=sampling_rate)
    print("Data loading complete.")
    print("Classifing between classes:", label_map)

    # Initialise cross validation
    test_size = 0.2
    outer_cv = LeaveOneGroupOut()
    single_split = StratifiedGroupKFold(
        n_splits=int(1/test_size), shuffle=True, random_state=RANDOM_SEED
        )

    # Initialise a dictionary to store overall results
    all_results = {
        "meta_data": {
            "model_name": model_name,
            "class_names": class_names,
            "label_map": label_map,
            "epoch_duration": epoch_duration,
            "sampling_rate": sampling_rate,
            "chans": chans,
            "samples": samples
        },
        "outer_folds": [],
        "subject_metadata": {}
    }

    # Model checkpoint and early stopping for outer folds
    result_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", model_name, "_".join(class_names))
    os.makedirs(result_dir, exist_ok=True)

    # Outer Leave-One-Subject-Out Cross-Validation for model evaluation
    for outer_fold, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(X_all, y_all, groups=subjects_all)
    ):
        print(f"\n--- Outer Fold {outer_fold + 1} ---")

        # Split train/validation and test sets
        X_train_val, y_train_val, subjects_train_val = (
            X_all[train_val_idx],
            y_all[train_val_idx],
            subjects_all[train_val_idx]
        )
        X_test, y_test, subjects_test = (
            X_all[test_idx],
            y_all[test_idx],
            subjects_all[test_idx]
        )
        test_subject_id = subjects_test[0]

        # Run hyperparameter optimisation with inner CV
        # to find the best configuration for this outer fold
        study = optuna.create_study(
            direction="maximize",
            study_name=f"Outer_Fold_{outer_fold + 1}_Study",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5))

        study.optimize(
            lambda trial: objective(
                trial, outer_fold, single_split, X_train_val, y_train_val,
                subjects_train_val, num_classes, chans, samples, class_names
                ),
            n_trials=10)

        # Retrieve the best model from the study
        best_trial = study.best_trial
        best_score = best_trial.value
        best_params = best_trial.params

        print(f"Best validation accuracy: {best_score}")
        print("Best hyperparameters: ")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # Retrain best model on entire train/val set
        # Build new model with best hyperparameters
        new_model = build_model(best_params, num_classes, chans, samples)
        best_epochs = int(round(best_trial.user_attrs["avg_epochs"]))

        new_model.fit(X_train_val, y_train_val,
                      epochs=best_epochs,
                      batch_size=best_params["batch_size"],
                      class_weight=calculate_class_weights(y_train_val),
                      verbose=1)

        # Save the best model weights for this outer fold
        model_filepath = os.path.join(
            result_dir, f"best_model_outer_fold_{outer_fold+1}.keras"
            )
        new_model.save(model_filepath)

        # Reload the best model weights
        new_model = tf.keras.models.load_model(model_filepath)

        # Evaluate the best configuration on the test set for this outer fold
        test_metrics = calculate_metrics(new_model, X_test, y_test)
        print(f"Outer fold {outer_fold+1} Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"    {metric_name}: {metric_value}")

        # Store outer fold results
        fold_result = {
            "outer_fold": outer_fold + 1,
            "test_subject_id": test_subject_id,
            "best_params": best_params,
            "best_epochs": best_epochs,
            "model_filepath": model_filepath,

            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],

            "true_labels": test_metrics["y_true"].tolist(),
            "pred_probs": test_metrics["y_pred_probs"].flatten().tolist(),
            "pred_labels": test_metrics["y_pred"].tolist()
            }

        all_results["outer_folds"].append(fold_result)

        # Clear session
        del new_model
        K.clear_session()
        gc.collect()

    with open(os.path.join(result_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print("LOSO cross-validation complete. Results saved.")
