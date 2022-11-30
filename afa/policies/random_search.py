import logging
import math
from typing import Callable, Optional, Tuple, Dict, Any, Union

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from jax.experimental import jax2tf
from tensorflow import saved_model
from tensorflow_probability.substrates.numpy.distributions import Categorical

from afa.policies.base import Policy
from afa.rewards import get_reward_fn
from afa.typing import Observation, NumpyDistribution, Array, ConfigDict


class RandomSearchPolicy(Policy):
    """Policy that executes a random search over subsets of features to acquire.

    Note that this is a "cheating" policy. It must be provided with the ground truth
    features and class label in order to evaluate candidate subsets of features that
    should be acquired in the future. The actual distribution that this policy
    returns is uniform over the features that were in the subset found by the search.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
        max_observed: The maximum percentage of features that can be marked as
            observed in the masks returned by the function.
        num_samples: The number of sample masks to generate in each search. The more
            samples that are used, the more likely it is for the search to find the
            optimal solution.
        fitness_fn: The fitness function that is used to score each candidate mask.
            This is a function that accepts `(x_o, b, y)`, where `y` is the ground
            truth target for the instance, and returns a scalar score. Higher scores
            are considered better. This function should be a pure Jax function.
        backend: The XLA backend to execute the search on. Can be either "cpu", "gpu",
            or "tpu".

    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        max_observed: float,
        num_samples: int,
        fitness_fn_config: ConfigDict,
        backend: Optional[str] = None,
    ):
        super().__init__(observation_space, action_space)

        assert action_space.n == math.prod(observation_space["mask"].shape) + 1, (
            "RandomSearchPolicy assumes an action space with only a single "
            "terminal action."
        )

        fitness_fn = create_fitness_fn(fitness_fn_config)

        search_fn = create_random_search_fn(
            max_observed=max_observed,
            num_samples=num_samples,
            fitness_fn=fitness_fn,
        )
        self._search_fn = jax.jit(search_fn, backend=backend)

        self._prng = hk.PRNGSequence(91)
        self._last_search_result = None

    def compute_policy(
        self, obs: Observation, **kwargs
    ) -> Tuple[NumpyDistribution, Dict[str, Any]]:
        assert "current_example" in kwargs, (
            "RandomSearchPolicy requires that the ground truth features be passed to "
            "compute_policy with the 'current_example' keyword argument."
        )
        assert "current_target" in kwargs, (
            "RandomSearchPolicy requires that the ground truth label be passed to "
            "compute_policy with the 'current_target' keyword argument."
        )

        if kwargs.get("reuse_search", False):
            assert (
                self._last_search_result is not None
            ), "Cannot reuse search result. None available."
            winning_mask = self._last_search_result
        else:
            winning_mask = np.asarray(
                self._search_fn(
                    self._prng.next(),
                    kwargs["current_example"],
                    obs["mask"],
                    kwargs["current_target"],
                )
            )
            self._last_search_result = winning_mask

        probs = np.zeros((self.action_space.n,), dtype=np.float32)

        if np.array_equal(obs["mask"], winning_mask):
            # If the search returns the same mask that we currently have, this means
            # that it believes we can't improve by acquiring any more features, so
            # we assign all probability to the terminal action.
            probs[-1] = 1.0
            self._last_search_result = None
        else:
            valid_actions = np.logical_xor(obs["mask"], winning_mask)
            valid_actions = np.reshape(valid_actions, [-1]).astype(np.float32)
            valid_actions /= np.sum(valid_actions)
            probs[:-1] = valid_actions

        with np.errstate(divide="ignore"):
            logits = np.where(probs == 0, -1e12, np.log(probs))

        return Categorical(logits=logits), {}


def create_classifier_fitness_fn(
    classifier_dir: str,
    terminal_reward_fn_config: ConfigDict,
    acquisition_cost: Union[float, Array],
) -> Callable[[jnp.ndarray, jnp.ndarray, int], float]:
    """Creates a fitness function that uses the provided classifier.

    This return fitness function assumes the basic active acquisition reward scheme
    where each acquisition has a cost associated with it and there is a final reward
    at the terminal action that is based on the classifier's current predictions.

    Args:
        classifier_dir: A path to a `SavedModel` that is the classifier.
        terminal_reward_fn_config: A config dict for the terminal reward function.
            Should include the keys "type" and "kwargs". See `get_reward_fn`.
        acquisition_cost: An array of the same shape as the environment masks where each
            element corresponds to the acquisition cost for the feature at that
            location. Or, a float if all features have the same cost.

    Returns:
        A function that accepts the features, mask and ground truth label, and returns
        a fitness score.
    """
    classifier = saved_model.load(classifier_dir)
    terminal_reward_fn = get_reward_fn(
        terminal_reward_fn_config["type"], **terminal_reward_fn_config.get("kwargs", {})
    )

    # Supress some expected warnings generated by the usage of jax2tf below.
    logging.root.setLevel(logging.ERROR)

    def fitness_fn(x_o: jnp.ndarray, mask: jnp.ndarray, y: int) -> float:
        logits = jax2tf.call_tf(classifier)(
            {"x": jnp.expand_dims(x_o, 0), "b": jnp.expand_dims(mask, 0)}
        )
        logits = jnp.squeeze(logits)
        terminal_reward = terminal_reward_fn(y, logits)
        if isinstance(acquisition_cost, float):
            acquisition_cost_ = jnp.ones_like(mask) * acquisition_cost
        else:
            acquisition_cost_ = acquisition_cost
        total_cost = jnp.sum(mask * acquisition_cost_)
        score = terminal_reward - total_cost
        return score

    return fitness_fn


def create_fitness_fn(
    config: ConfigDict,
) -> Callable[[jnp.ndarray, jnp.ndarray, int], float]:
    """Creates a fitness function from a config dict.

    Args:
        config: The fitness function config dict. This should contain the keys
            "type" and "kwargs".

    Returns:
        A fitness function.
    """
    fitness_type = config["type"]

    if fitness_type == "classifier":
        return create_classifier_fitness_fn(**config["kwargs"])

    raise ValueError(f"'{fitness_type}' is not a valid fitness function type.")


def create_generate_mask_fn(
    max_observed: float,
) -> Callable[[PRNGKey, jnp.ndarray], jnp.ndarray]:
    """Creates a function that generates uniformly random masks.

    Masks are created based on a "current" mask with is passed into the function.
    Given a current mask, which has some observed values, the function will return
    a new mask such that the same values are still observed, but some of the previously
    unobserved values may now be marked as observed. The total number of observed values
    in the new mask will now be more than `max_observed` percent of the total
    number of values though.

    Args:
        max_observed: The maximum percentage of features that can be marked as
            observed in the masks returned by the function.

    Returns:
        A function that accepts a Jax `PRNGKey` and a random mask, and returns a new
        random mask with additional features randomly masked out.
    """

    # This function is only for generating the initial mask, when nothing is observed.
    def _generate_initial_mask(key: PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
        mask_shape = mask.shape
        d = math.prod(mask_shape)
        h = int(round(d * max_observed))
        q = jax.random.choice(key, h)
        inds = jax.random.permutation(key, jnp.arange(d))
        new_mask = jnp.where(inds <= q, 1.0, 0.0)
        new_mask = jnp.reshape(new_mask, mask_shape)
        return new_mask

    # This function generates a mask in the general case, when things have already
    # been observed. It is currently much slower than the above version though.
    def _generate_mask_from_observed(key: PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
        flat_mask = jnp.reshape(mask, [-1])

        # The total number of features.
        d = flat_mask.shape[-1]
        # The upper bound on the number of features that we want to be marked.
        h = int(round(d * max_observed))

        noise = jax.random.uniform(key, shape=flat_mask.shape)
        inds = jnp.argsort(jnp.where(flat_mask == 1, 2, noise))

        # This is the number of features that are already observed.
        o = jnp.count_nonzero(flat_mask)
        # This is the number of new features that will be marked as observed in the new
        # mask. This will be a number in the range [1, h - o]. Note that we don't need
        # to consider masks where 0 features become marked.
        q = 1 + jnp.argmax(jnp.where(jnp.arange(d) < h - o, noise, -1))

        def loop_fn(i, cur_mask):
            cur_mask = cur_mask.at[inds[i]].set(1.0)
            return cur_mask

        new_mask = jax.lax.fori_loop(0, q, loop_fn, flat_mask)
        return jnp.reshape(new_mask, mask.shape)

    def generate_mask(key: PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
        index = jnp.max(mask).astype(jnp.int32)
        branches = [_generate_initial_mask, _generate_mask_from_observed]
        # Choose which version of the mask generation function to use based on whether
        # anything has been observed yet.
        return jax.lax.switch(index, branches, key, mask)

    return generate_mask


def create_random_search_fn(
    max_observed: float,
    num_samples: int,
    fitness_fn: Callable[[jnp.ndarray, jnp.ndarray, int], float],
) -> Callable[[PRNGKey, jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    """Creates a pure Jax function that executes a random search over masks.

    The random search will uniformly generate random masks and score each one based on
    the provided fitness function. It is assumed that a mask's fitness is a function
    of the currently observed features and the ground truth target for the current
    instance. The search will output the mask that achieves the highest fitness score.
    Generated masks will always retain the same observed values of the current mask
    that is passed into the search function.

    Note that the returned function assumes that just a single data point and label
    are provided, i.e. that there is no batch dimension. The function can be vmap'ed
    to allow for processing of batches.

    Args:
        max_observed: The maximum percentage of features that can be marked as
            observed in the masks returned by the function.
        num_samples: The number of random masks to generate and score during the
            search.
        fitness_fn: A function that accepts `(x_o, b, y)`, where `y` is the ground
            truth target for the instance, and returns a scalar score. This function
            should be a pure Jax function.

    Returns:
        A function that accepts a Jax `PRNGKey`, the ground truth features, and the
        ground truth target, performs a random search for the mask that maximizes the
        fitness function, and returns the winning mask.
    """
    generate_mask = create_generate_mask_fn(max_observed)

    def search_fn(
        key: PRNGKey, x: jnp.ndarray, mask: jnp.ndarray, y: int
    ) -> jnp.ndarray:
        def run_trial(_, state):
            key, best_score, best_mask = state
            next_key, mask_key = jax.random.split(key, 2)

            candidate_mask = generate_mask(mask_key, mask)
            x_o = x * candidate_mask
            score = fitness_fn(x_o, candidate_mask, y)

            best_mask = jnp.where(score > best_score, candidate_mask, best_mask)
            best_score = jnp.maximum(score, best_score)

            return next_key, best_score, best_mask

        initial_score = fitness_fn(x * mask, mask, y)
        _, _, best_mask = jax.lax.fori_loop(
            0, num_samples, run_trial, (key, initial_score, mask)
        )

        return best_mask

    return search_fn
