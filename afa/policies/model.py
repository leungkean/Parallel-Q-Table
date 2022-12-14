from typing import Tuple, Dict, Any

import gym
import numpy as np
import tensorflow as tf
import tree
from ray.rllib import SampleBatch
from tensorflow_probability.substrates.numpy.distributions import Categorical

from afa.networks.models.base import Model
from afa.networks.models.utils import build_model
from afa.policies.base import Policy
from afa.typing import Observation, NumpyDistribution, Weights


class ModelPolicy(Policy):
    """A policy generated by a learned model.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
        model_config: The config dict for the model. This will be passed to
            `build_model`.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        model_config: Dict[str, Any],
    ):
        super().__init__(observation_space, action_space)
        self._model = build_model(observation_space, action_space, model_config)
        self._forward = tf.function(
            lambda obs: self._model(
                tree.map_structure(lambda x: tf.expand_dims(x, 0), obs)
            )
        )

        # Call the model with dummy inputs to force the graph to be built.
        self._forward(observation_space.sample())

    @property
    def model(self) -> Model:
        return self._model

    def compute_policy(
        self, obs: Observation, **kwargs
    ) -> Tuple[NumpyDistribution, Dict[str, Any]]:
        pi, values = self._forward(obs)

        extra_info = {}

        if values is not None:
            extra_info[SampleBatch.VF_PREDS] = np.squeeze(values, 0)

        return Categorical(np.squeeze(pi.logits, 0)), extra_info

    def get_weights(self) -> Weights:
        return self._model.get_weights()

    def set_weights(self, weights: Weights):
        self._model.set_weights(weights)
