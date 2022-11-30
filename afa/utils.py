from typing import NamedTuple, Optional, List, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from afa.typing import ConfigDict, Array


def get_schedule(config: ConfigDict) -> optax.Schedule:
    """Gets a schedule function from a config dict.

    Args:
        config: The configuration for the schedule. Should contain the key "type" and
            the key "kwargs", the latter of which will be passed to the specified
            schedule.

    Returns:
        A `optax.Schedule`.
    """
    if config["type"] == "constant":
        return optax.constant_schedule(**config["kwargs"])

    if config["type"] == "linear":
        return optax.linear_schedule(**config["kwargs"])

    if config["type"] == "polynomial":
        return optax.polynomial_schedule(**config["kwargs"])

    if config["type"] == "exponential":
        return optax.exponential_decay(**config["kwargs"])

    raise ValueError(f"'{config['type']}' is not a valid schedule type.")


def unobserved_mse(x: Array, b: Array, preds: Array) -> Array:
    """Computes the MSE of unobserved features.

    IMPORTANT: This function assumes the inputs are only for a single instance.
    To compute MSEs for a batch of data, VMAP this function.

    Args:
        x: The ground truth values.
        b: The bitmask indicating which features are observed.
        preds: The predicted values.

    Returns:
        The MSE over the unobserved features.
    """
    se = (x - preds) ** 2 * (1 - b)
    u = jnp.count_nonzero(1 - b)
    mse = jnp.where(u == 0, 0, se / u)
    return mse


def _weight_decay_exclude(
    exclude_names: Optional[List[str]] = None,
) -> Callable[[str, str, jnp.ndarray], bool]:
    """Logic for deciding which parameters to include for weight decay..

    Args:
      exclude_names: an optional list of names to include for weight_decay. ['w']
        by default.

    Returns:
      A predicate that returns True for params that need to be excluded from
      weight_decay.
    """
    # By default weight_decay the weights but not the biases.
    if not exclude_names:
        exclude_names = ["b"]

    def exclude(module_name: str, name: str, value: jnp.array):
        del value
        # Do not weight decay the parameters of normalization blocks.
        if any([norm_name in module_name for norm_name in ["layer_norm", "batchnorm"]]):
            return True
        else:
            return name in exclude_names

    return exclude


class AddWeightDecayState(NamedTuple):
    """Stateless transformation."""


def add_weight_decay(
    weight_decay: float, exclude_names: Optional[List[str]] = None
) -> optax.GradientTransformation:
    """Add parameter scaled by `weight_decay` to the `updates`.

    Same as optax.add_decayed_weights but can exclude parameters by name.

    Args:
      weight_decay: weight_decay coefficient.
      exclude_names: an optional list of names to exclude for weight_decay. ['b']
        by default.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return AddWeightDecayState()

    def update_fn(updates, state, params):
        exclude = _weight_decay_exclude(exclude_names=exclude_names)

        u_ex, u_in = hk.data_structures.partition(exclude, updates)
        _, p_in = hk.data_structures.partition(exclude, params)
        u_in = jax.tree_map(lambda g, p: g + weight_decay * p, u_in, p_in)
        updates = hk.data_structures.merge(u_ex, u_in)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
