from typing import Callable

import jax
import jax.numpy as jnp

from afa.typing import Array


def get_cross_entropy_reward_fn(
    temperature: float = 1.0,
) -> Callable[[int, Array], float]:
    """Creates a terminal reward function that uses the cross-entropy loss.

    Args:
        temperature: A temperature to be applied to the logits before the
            cross-entropy is computed.

    Returns:
        A reward function that accepts the ground truth class label and the
        classifier logits.
    """

    def reward_fn(label: int, logits: Array) -> float:
        logits = logits / temperature
        return jnp.sum(
            jax.nn.one_hot(label, logits.shape[0]) * jax.nn.log_softmax(logits),
        )

    return reward_fn


def get_reward_fn(reward_type: str, **kwargs) -> Callable[[int, Array], float]:
    """Gets a reward function of a particular type.

    Args:
        reward_type: The type of reward function to create.
        **kwargs: Keyword arguments to be passed to the creation function for the
            particular reward type.

    Returns:
        A reward function.
    """
    if reward_type == "xent":
        return get_cross_entropy_reward_fn(kwargs.get("temperature", 1.0))

    raise ValueError(f"{reward_type} is not a valid reward type.")
