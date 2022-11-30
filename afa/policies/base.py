import abc
from typing import Tuple, Dict, Any

import gym

from afa.typing import Observation, NumpyDistribution, Weights


class Policy(abc.ABC):
    """An abstract base class that represents policies.

    A policy is for computing the distribution over actions for a given state in a
    particular environment.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def compute_policy(
        self, obs: Observation, **kwargs
    ) -> Tuple[NumpyDistribution, Dict[str, Any]]:
        """Computes the distribution over actions for the given observation.

        Args:
            obs: The environment observation to compute the policy for.
            **kwargs: Additional keyword arguments that can provided additional
                information which may be needed for certain subclasses.

        Returns:
            pi: A `NumpyDistribution` representing the current policy.
            extra_info: A (possibly empty) dictionary containing extra information
                from the policy that may be useful.
        """
        pass

    def get_weights(self) -> Weights:
        """Gets any weights used by this policy.

        This function must be overridden in subclasses that use weights.

        Returns:
            The policy's weights.
        """
        return []

    def set_weights(self, weights: Weights):
        """Sets the weights of this policy.

        This function must be overridden in subclasses that use weights.

        Args:
            weights: The new weights for the policy.

        Returns:
            None
        """
        pass
