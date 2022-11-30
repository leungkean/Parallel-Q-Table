from typing import Union, Iterable, Mapping, Any, List, Dict

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.numpy import distributions as tfdnp

Scalar = Union[float, int]
Array = Union[np.ndarray, jnp.ndarray]
Numeric = Union[Array, Scalar]
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
NumericTree = Union[Numeric, Iterable["NumericTree"], Mapping[Any, "NumericTree"]]

Observation = ArrayTree

NumpyDistribution = tfdnp.Distribution

Weights = List[np.ndarray]

ConfigDict = Dict[str, Any]
