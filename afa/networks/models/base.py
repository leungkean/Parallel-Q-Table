import abc
from typing import List, Tuple, Optional

import gym
from tensorflow import Tensor
from tensorflow_probability.python.distributions import Distribution

from afa.typing import Observation, Weights


class Model(abc.ABC):
    """Base class for models.

    A Model represents any networks used by an agent that have trainable parameters.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.spaces.Discrete):
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    @abc.abstractmethod
    def trainable_weights(self) -> List[Tensor]:
        """The model's trainable weights."""
        pass

    @abc.abstractmethod
    def __call__(
        self, obs: Observation, training: bool = False
    ) -> Tuple[Distribution, Optional[Tensor]]:
        """Performs a forward pass of the model.

        This function should return a `tfd.Distribution`, representing the policy.

        Args:
            obs: The observations from the environment.
            training: Whether or not the model should be in training mode.

        Returns:
            pi: A `Distribution`, representing the policy.
            value_preds: A tensor containing value predictions, or None if the model
                does not have a value head.
        """
        pass

    @abc.abstractmethod
    def get_weights(self) -> Weights:
        """Gets any weights used by this model.

        Returns:
            The model's weights.
        """
        return []

    @abc.abstractmethod
    def set_weights(self, weights: Weights):
        """Sets the weights of this model.

        Args:
            weights: The new weights for the model.

        Returns:
            None
        """
        pass
