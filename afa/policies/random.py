import math
from typing import Tuple, Dict, Any

import gym
import numpy as np
from tensorflow_probability.substrates.numpy.distributions import Categorical

from afa.policies.base import Policy
from afa.typing import Observation, NumpyDistribution


class RandomPolicy(Policy):
    """A policy that selects among valid actions completely at random."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space, action_space)

        # The number of actions that are non-terminal, i.e. for acquiring a feature.
        self._num_nonterminal = math.prod(self.observation_space["mask"].shape)

    def compute_policy(
        self, obs: Observation, **kwargs
    ) -> Tuple[NumpyDistribution, Dict[str, Any]]:
        logits = np.zeros((self.action_space.n,), dtype=np.float32)
        # Mask out already acquired features.
        logits[: self._num_nonterminal] -= obs["mask"] * 1e12
        return Categorical(logits=logits), {}
