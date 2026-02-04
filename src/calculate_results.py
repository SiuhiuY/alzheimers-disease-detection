import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, recall_score,
    auc, roc_auc_score, roc_curve
)


def append_meta_data(results, meta):
    """
    Append metadata to results dictionary.

    Args:
        results (dict): Dictionary containing model results with subject IDs.

    Returns:
        dict: Updated results dictionary with appended metadata.
    """
    results["subject_metadata"] = {}

    for fold in results["outer_folds"]:
        for subject_id in np.unique(fold["test_subject_id"]):
            subject_meta = meta[meta["participant_id"] == subject_id].iloc[0]
            results["subject_metadata"][subject_id] = {
                "gender": subject_meta["Gender"],
                "age": subject_meta["Age"],
                "group": subject_meta["Group"]
            }
    return results


def get_all_accuracy(results):
    """
    Compile test accuracies across all outer folds.

    Args:
        results (dict): Dictionary containing outer fold results
        with true labels and predicted labels.

    Returns:
        accuracies (list): List of test accuracies for each outer fold.
    """
    accuracies = []

    for fold in results["outer_folds"]:
        acc = fold["test_accuracy"]
        subject_id = fold["test_subject_id"]
        group = results["subject_metadata"][subject_id]["group"]

        accuracies.append({
            "test_accuracy": acc,
            "test_subject_id": subject_id,
            "subject_group": group
        })

    return accuracies


def calculate_all_metrics(results):
    """
    Calculate various performance metrics for model results.

    Args:
        results (dict): Dictionary containing outer fold results
        with true labels and predicted labels.

    Returns:
        metrics (dict): Dictionary of performance metrics.
    """
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold in results["outer_folds"]:
        all_y_true.extend(fold["test_true_labels"])
        all_y_pred.extend(fold["test_pred_labels"])
        all_y_prob.extend(fold["test_pred_probs"])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    accuracy = accuracy_score(all_y_true, all_y_pred)
    f1_score_value = f1_score(all_y_true, all_y_pred)
    sensitivity = recall_score(all_y_true, all_y_pred, pos_label=1)
    specificity = recall_score(all_y_true, all_y_pred, pos_label=0)
    roc_auc = roc_auc_score(all_y_true, all_y_prob)
    cm = confusion_matrix(all_y_true, all_y_pred)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1_score_value,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

    return metrics


def get_all_roc(results):
    """
    Compile ROC curve data across all outer folds.

    Args:
        results (dict): Dictionary containing outer fold results
        with true labels and predicted probabilities.

    Returns:
        fpr (np.array): False positive rates.
        tpr (np.array): True positive rates.
        auc_val (float): Area under the ROC curve.
    """
    all_y_true = []
    all_y_prob = []

    for fold in results["outer_folds"]:
        all_y_true.extend(fold["true_labels"])
        all_y_prob.extend(fold["pred_probs"])

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
    auc_val = auc(fpr, tpr)

    return fpr, tpr, auc_val
