from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from torch.utils.data import Subset
import numpy as np

# Reference:
# https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit


class CrossValidator:
    """
    Custom Cross-Validation supporting Nested Leave-One-Subject-Out (LOSO),
    Nested Stratified Group K-Fold (SGKF) and single train-test split.
    """

    def __init__(
        self, features, labels, subjects, cv_strategy=None,
        n_splits=None, test_size=None, shuffle=False, random_state=None
    ):
        """Initialize the CrossValidator.

        Args:
            features (np.ndarray): Feature data.
            labels (np.ndarray): Corresponding class labels.
            subjects (np.ndarray): Corresponding subject IDs.
            cv_strategy (str): Cross-validation strategy
                ('loso', 'sgkf', or None). If None, performs a single
                train-test split.
            n_splits (int, optional): Number of splits
                for Stratified Group K-Fold.
            shuffle (bool): Whether to shuffle data before splitting
                (only for Stratified Group K-Fold).
            random_state (int, optional): Random seed for shuffling.
        """
        self.features = features
        self.labels = labels
        self.subjects = subjects
        self.n_splits = n_splits
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv_strategy = cv_strategy

        # Validate parameters
        if self.cv_strategy is None:
            if self.test_size is None:
                raise ValueError(
                    "test_size must be specified for single train-test split."
                )
            if not (0 < self.test_size < 1):
                raise ValueError("test_size must be between 0 and 1.")

        elif self.cv_strategy.lower() == 'sgkf':
            if self.n_splits is None or self.n_splits < 2:
                raise ValueError(
                    "n_splits must be at least 2 for Stratified Group K-Fold."
                )

        # Initialise the appropriate splitter
        self.splitter = self.get_splitter()

        assert len(self.features) == len(self.labels) == len(self.subjects), \
            "Features, labels, and subjects must have the same length."

    def get_splitter(self):
        """Get the appropriate splitter based on the cv_strategy."""
        if self.cv_strategy is None:
            print("Using single train-test split.")
            single_splitter = StratifiedGroupKFold(
                n_splits=int(1/self.test_size),
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return single_splitter

        elif self.cv_strategy.lower() == 'loso':
            print("CV Strategy: Leave One Subject Out (LOSO)")
            return LeaveOneGroupOut()

        elif self.cv_strategy.lower() == 'sgkf':
            print("CV Strategy: Stratified Group K-Fold (SGKF)")
            return StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )

        else:
            raise ValueError(
                f"Unsupported cv_strategy: {self.cv_strategy}. "
                f"Choose 'loso' or 'sgkf'."
            )

    def cv_loop(self):
        """Generate data split indices at each fold."""
        if self.cv_strategy is None:
            # Single train-test split
            train_idx, test_idx = next(self.splitter.split(
                self.features, self.labels, self.subjects))
            if len(set(np.unique(self.subjects[train_idx])) &
                   set(np.unique(self.subjects[test_idx]))) != 0:
                raise ValueError(
                    "Subjects overlap between train and test sets!")
            yield 0, train_idx, test_idx

        else:
            # Multiple folds
            for i, (train_idx, test_idx) in enumerate(
                self.splitter.split(
                    self.features, self.labels, self.subjects)):
                if len(set(np.unique(self.subjects[train_idx])) &
                       set(np.unique(self.subjects[test_idx]))) != 0:
                    raise ValueError(
                        f"Subjects overlap between train and test sets in "
                        f"cv fold {i}."
                    )
                yield i, train_idx, test_idx  # yield one fold at a time
