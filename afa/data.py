from typing import Tuple

import numpy as np
import tensorflow_datasets as tfds


def load_supervised_split_as_numpy(
    dataset: str, split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a supervised TensorFlow Dataset as a numpy array.

    Args:
        dataset: The name of the dataset.
        split: The split to load.

    Returns:
        The data as the tuple `(x, y)`.
    """
    ds = tfds.load(dataset, split=split, as_supervised=True)
    x, y = ds.batch(ds.cardinality().numpy()).get_single_element()
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x.shape) == 4:
        x = x.astype(np.float32) / 255.0
    return x, y


def load_unsupervised_split_as_numpy(dataset: str, split: str) -> np.ndarray:
    """Loads an unsupervised TensorFlow Dataset as a numpy array.

    Args:
        dataset: The name of the dataset.
        split: The split to load.

    Returns:
        The data as the tuple `(x, y)`.
    """
    ds = tfds.load(dataset, split=split)
    x = ds.batch(ds.cardinality().numpy()).get_single_element()
    data_key = "features" if "features" in x else "image"
    x = x[data_key]
    x = np.asarray(x)
    if len(x.shape) == 4:
        x = x.astype(np.float32) / 255.0
    return x
