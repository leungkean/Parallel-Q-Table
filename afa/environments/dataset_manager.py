import random
from typing import Optional, Tuple

import numpy as np

from afa.typing import ArrayTree


class EnvironmentDatasetManager:
    """An actor responsible for providing instances to acquisition environments.

    This class can be thought of as a queue, which lives in its own process, that
    provides data instances to active acquisition environments. Whenever an environment
    is reset (i.e. is ready to start a new episode), it asks an instance of this class
    for a new example from the dataset, which it will then use for the new episode.

    There are two main reasons for the existence of this class:
        1) We generally run many copies of the environment in parallel when training or
           even when evaluating. If the entire dataset were to be stored in each copy
           of the environment, then we would have many copies of a potentially very
           large dataset being (needlessly) created, therefore consuming large
           amounts of memory.
        2) This class provides more control over which instances are being sent to
           the environments and how many "epochs" have elapsed. With this class, we can
           run many environments in parallel, but still ensure that e.g. each instance
           in the test set is used exactly once.

    Args:
        features: A dataset in the form of a NumPy array. The first dimension should be
            the batch dimension. Note that only data features should be included here --
            if using a classification dataset with targets, the targets should be
            provided separately via the `targets` keyword argument.
        targets: Optional NumPy array containing classification targets, which should
            be aligned with `features`. If this is `None`, then calls to
            `get_new_instance` will only return a single item: the features. If
            targets are provided, then calls to `get_new_instance` will return both
            the features and corresponding target. It is assumed that the provided
            targets should range from 0 to n - 1, where n is the number of classes.
        error_on_new_epoch: If True, then an error will be raised if this manager
            tries to move to a second epoch (i.e. if `get_new_instance` is called
            enough times that repeat instances would need to be used). This can
            be useful for e.g. ensuring that instances are not getting evaluated
            multiple times at test time.
        seed: Optionally specified random seed to be used by this manager.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        error_on_new_epoch: bool = False,
        seed: Optional[int] = None,
    ):
        self._features = np.asarray(features, np.float32)
        self._targets = None if targets is None else np.asarray(targets, np.int32)
        self._error_on_new_epoch = error_on_new_epoch

        self._prng = np.random.RandomState(seed or random.randrange(int(2e9)))

        self._current_index = len(self._features)
        self._current_permutation = None
        self._num_epochs = -1
        self._num_queries = 0

    def features_shape(self) -> Tuple[int, ...]:
        """Returns the shape of a single instance."""
        return self._features.shape[1:]

    def targets_shape(self) -> Tuple[int, ...]:
        """Returns the shape of a single target."""
        if len(self._targets) == 1:
            return (1,)
        return self._targets.shape[1:]

    def num_classes(self) -> Optional[int]:
        """Returns the number of classes (or None if not using targets)."""
        if self._targets is None:
            return None

        return np.max(self._targets) + 1

    def num_targets(self) -> Optional[int]:
        if self._targets is None:
            return None
        elif len(self._targets.shape) == 1:
            return 1

        return self._targets.shape[1]

    def lifetime_epochs(self) -> int:
        """Returns the total number of epochs that have elapsed."""
        return self._num_epochs

    def lifetime_queries(self) -> int:
        """Returns the total number of times this manager has been queried."""
        return self._num_queries

    def get_new_instance(self) -> ArrayTree:
        """Gets a new instance from the dataset.

        Returns:
            If `targets` was not specified, then a single array will be returned
            that contains the features. If `targets` are being used, then
            the corresponding class label for the returned features will also be
            returned.
        """
        self._current_index += 1

        if self._current_index >= len(self._features):
            self._current_index = 0
            self._current_permutation = self._prng.permutation(
                np.arange(len(self._features))
            )
            self._num_epochs += 1

            if self._error_on_new_epoch and self._num_epochs > 0:
                raise RuntimeError("EnvironmentDatasetManager has started a new epoch.")

        self._num_queries += 1

        if self._targets is not None:
            features = self._features[self._current_permutation[self._current_index]]
            targets = self._targets[self._current_permutation[self._current_index]]
            return features, targets

        return self._features[self._current_permutation[self._current_index]]
