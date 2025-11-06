from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from torch.utils.data import Subset
import numpy as np


class CrossValidator:
    """
    Custom Cross-Validation supporting Nested Leave-One-Subject-Out (LOSO)
    and Nested Stratified Group K-Fold (SGKF).
    """

    def __init__(
        self, full_dataset, features, labels, subjects, cv_strategy,
        n_splits=10, shuffle=False, random_state=None
    ):
        """Initialize the CrossValidator.

        Args:
            data (np.ndarray): Feature data.
            labels (np.ndarray): Corresponding class labels.
            subjects (np.ndarray): Corresponding subject IDs.
            cv_strategy (str): Cross-validation strategy
                ('Stratified Group K-Fold' or 'Leave-One-Group-Out').
            n_splits (int): Number of splits for Stratified Group K-Fold.
            shuffle (bool): Whether to shuffle data before splitting
                (only for Stratified Group K-Fold).
            random_state (int, optional): Random seed for shuffling.
        """
        self.data = full_dataset
        self.features = features
        self.labels = labels
        self.subjects = subjects
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.splitter = self.get_splitter(cv_strategy)

        assert len(self.features) == len(self.labels) == len(self.subjects), \
            "Features, labels, and subjects must have the same length."

    def get_splitter(self, cv_strategy):
        """Get the appropriate splitter based on the cv_strategy."""
        if cv_strategy.lower() == 'loso':
            print("CV Strategy: Leave One Subject Out (LOSO)")
            return LeaveOneGroupOut()
        elif cv_strategy.lower() == 'sgkf':
            print("CV Strategy: Stratified Group K-Fold (SGKF)")
            return StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unsupported cv_strategy: {cv_strategy}. "
                f"Choose 'loso' or 'sgkf'."
            )

    def inner_loop(self):
        """Generate train-test indices at each fold for hyperparameter
        tuning."""
        for i, (train_idx, test_idx) in enumerate(
            self.splitter.split(
                self.features, self.labels, self.subjects)):
            if len(set(np.unique(self.subjects[train_idx])) &
                   set(np.unique(self.subjects[test_idx]))) != 0:
                raise ValueError(
                    f"Subjects overlap between train and test sets in "
                    f"inner fold {i}."
                )
            yield i, train_idx, test_idx  # yield one fold at a time

    def outer_loop(self):
        """Generate train-test subsets at each fold for model evaluation."""
        for i, (train_idx, test_idx) in enumerate(
            self.splitter.split(
                self.features, self.labels, self.subjects)):
            if len(set(np.unique(self.subjects[train_idx])) &
                   set(np.unique(self.subjects[test_idx]))) != 0:
                raise ValueError(
                    f"Subjects overlap between train and test sets in "
                    f"outer fold {i}."
                )
            train_set = Subset(self.data, train_idx)
            test_set = Subset(self.data, test_idx)
            yield i, train_set, test_set
