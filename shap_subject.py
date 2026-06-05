import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.feature_loader import load_features
from src.util import reproducability
from src.models.pretrained_adaptive import PretrainedResNet18

# Reference:
# https://github.com/surnfnogl/ST_SHAP_code/blob/main/shapdatashow/shap_explain_mean.py


class DummyTrial:
    """
    A dummy trial class to mimic Optuna's trial interface for model loading.

    This class allows to create a trial object with predefined parameters,
    which can be used to initialise our model with the same architecture as
    the one used during training.
    """
    def __init__(self, params):
        """Initialise the DummyTrial with a dictionary of parameters.

        Args:
            params (dict): A dictionary where keys are parameter names and
            values are the corresponding parameter values to return.
        """
        self.params = params

    def _get_param(self, name):
        """Helper method to retrieve a parameter value from the params dictionary.

        Args:
            name (str): The name of the parameter to retrieve.

        Raises:
            KeyError: If the parameter name is not found in the dictionary.
        
        Returns:
            The value of the parameter from self.params if it exists.
        """
        if name not in self.params:
            raise KeyError(f"Missing parameters: {name}")
        return self.params[name]

    def suggest_categorical(self, name, choices):
        """Mimic Optuna's suggest_categorical method.

        Args:
            name (str): The name of the parameter to suggest.
            choices (list): A list of possible categorical choices.
        
        Returns:
            The value from self.params corresponding to "name" if it exists
        """
        return self._get_param(name)

    def suggest_int(self, name, low, high):
        """Mimic Optuna's suggest_int method.

        Args:
            name (str): The name of the parameter to suggest.
            low (int): The lower bound of the integer range.
            high (int): The upper bound of the integer range.

        Returns:
            The value from self.params corresponding to "name" if it exists
        """
        return self._get_param(name)

    def suggest_float(self, name, low, high, log=False):
        """Mimic Optuna's suggest_float method.

        Args:
            name (str): The name of the parameter to suggest.
            low (float): The lower bound of the float range.
            high (float): The upper bound of the float range.
            log (bool): Whether to sample in log space.

        Returns:
            The value from self.params corresponding to "name" if it exists
        """
        return self._get_param(name)


def load_model(best_params, input_shape, model_class, model_path, device):
    """Load the model with the given parameters and trained weights.

    Args:
        best_params (dict): The best parameters for the model.
        input_shape (tuple): The shape of the input data (C, H, W).
        model_class (class): The choice of model to instantiate.
        model_path (str): The path to the trained model file.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model ready for inference.
    """
    trial = DummyTrial(best_params)
    model = model_class(trial, input_shape=input_shape, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main(model_name, model_params, model_path, device, output_dir, subject_id,
         sampling_size, label_map, band, task_name, task_title, seed):
    """Main function to perform SHAP analysis on a single subject.

    Args:
        model_name (str): The name of the model architecture to use.
        model_params (dict): The parameters to initialise the model with.
        model_path (str): The path to the trained model file.
        device (torch.device): The device to perform computations on.
        output_dir (str): The directory to save SHAP results and plots.
        subject_id (int): The ID of the subject to explain.
        sampling_size (int): The number of background samples to use.
        label_map (dict): The mapping of labels to integers.
        band (str): The band of the data to use.
        task_name (str): The name of the task for labeling plots.
        task_title (str): The title of the task for labeling plots.
        seed (int): The random seed for reproducability.
    """
    # Set random seed
    reproducability(seed)

    # Create SHAP output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    features, labels, subjects = load_features(label_map=label_map, band=band)
    C, H, W = features.shape[1:]
    input_shape = (C, H, W)
    print("Data loaded:", features.shape)
    print(f"Available subjects: {np.unique(subjects)}")

    # Select one subject
    subject_ids = np.unique(subjects)
    subject_ind = subject_ids[subject_id]
    print("Selected subject:", subject_ind)

    # Choose the data for the selected subject
    to_explain = features[subjects == subject_ind]

    # Sample a background subset
    background_pool = features[subjects != subject_ind]
    sampling_mask = np.random.choice(len(background_pool),
                                     size=min(sampling_size,
                                              len(background_pool)),
                                     replace=False)
    background = background_pool[sampling_mask]

    # Convert to torch tensors for SHAP
    to_explain = torch.tensor(to_explain, dtype=torch.float32).to(device)
    background = torch.tensor(background, dtype=torch.float32).to(device)

    # Load model
    print("Loading model...")
    model = load_model(best_params=model_params, input_shape=input_shape,
                       model_class=model_name, model_path=model_path,
                       device=device)

    # SHAP
    explainer = shap.GradientExplainer(model, background)
    shap_values, _ = explainer.shap_values(to_explain, ranked_outputs=1)
    print("SHAP values computed:", shap_values.shape)

    # Remove output dimension
    # (n_windows, n_channels, 32, 32, 1) -> (n_windows, n_channels, 32, 32)
    shap_map = shap_values[:, :, :, :, 0]

    # Average across temporal channels
    # (n_windows, n_channels, 32, 32) -> (n_windows, 32, 32)
    # Then average across windows to get a single importance map
    # (n_windows, 32, 32) -> (32, 32)
    shap_map = shap_map.mean(axis=1).mean(axis=0)

    # Apply the same averaging to the input data for visualisation
    input_map = to_explain.cpu().numpy().mean(axis=1).mean(axis=0)

    # Save .npy
    combined = np.stack([input_map, shap_map], axis=0)
    subject_str = str(subject_ind)
    np.save(
        os.path.join(output_dir, f"{task_name}_{subject_str}.npy"), combined
        )

    # Plot
    vmax = np.percentile(np.abs(shap_map), 99)  # ignore extreme outliers
    vmin = -vmax  # symmetric color scale

    plt.figure(figsize=(4, 4))
    plt.imshow(input_map, cmap="gray", alpha=0.5)
    plt.imshow(shap_map, cmap="RdBu_r", alpha=0.6, vmin=vmin, vmax=vmax)
    plt.colorbar(label="SHAP value")
    plt.title(f"{subject_str} – {task_title}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{task_name}_{subject_str}.png"),
        dpi=300)
    plt.close()

    print("SHAP subject-level plot saved to:", output_dir)


if __name__ == "__main__":
    # Define configuration
    BAND = "alpha"
    TASKS = {
        "AD_CN": {
            "label_map": {"A": 1, "C": 0},
            "task_title": "AD vs CN",
            "task_name": "AD_CN",
        },
        "FTD_CN": {
            "label_map": {"F": 1, "C": 0},
            "task_title": "FTD vs CN",
            "task_name": "FTD_CN",
        },
    }
    task = TASKS["FTD_CN"]
    label_map = task["label_map"]
    task_title = task["task_title"]
    task_name = task["task_name"]
    MODEL_PATH = "results/PretrainedResNet18/alpha/C_F/best_model_outer_fold_31.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "shap_results"

    # Model fixed params from best trial
    BEST_PARAMS = {
        "resnet_fc_units": 256,
        "resnet_fc_dropout": 0.10812239313686567
    }

    model_name = PretrainedResNet18
    RANDOM_SEED = 123
    SUBJECT_INDEX = 48

    # Run SHAP analysis
    main(
        model_name=model_name,
        model_params=BEST_PARAMS,
        model_path=MODEL_PATH,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        subject_id=SUBJECT_INDEX,
        label_map=label_map,
        task_name=task_name,
        task_title=task_title,
        band=BAND,
        seed=RANDOM_SEED,
        sampling_size=400
        )
