from typing import Sequence, Optional, List, Tuple

import gym
import keras
from keras import layers
from tensorflow import Tensor
from tensorflow.python.ops.distributions.distribution import Distribution
from tensorflow_probability.python.distributions import Categorical

from afa.networks.models.base import Model
from afa.networks.models.utils import create_input_tensors, maybe_mask_logits
from afa.typing import Observation, Weights


class MLPModel(Model):
    """A simple multi-layer perceptron model.

    A value network can optionally be included as either a copy of the policy network
    or a separate head that shares a body with the policy network.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
        hidden_units: A sequence of integers, where the ith value is the number of
            units in the ith hidden layer.
        activation: The activation function to use.
        value_network: If None, then the model does not have a value head (and the
            values output of the model will be None). If "copy", then the value net
            will be a copy of the policy net and no weights are shared between the two.
            If "shared", then the two networks share a body and there are two separate
            output layers for the policy and values.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        hidden_units: Sequence[int] = (64, 64),
        activation: str = "relu",
        value_network: Optional[str] = None,
    ):
        super().__init__(observation_space, action_space)

        assert value_network in {None, "copy", "shared"}

        input_tensors, h = create_input_tensors(observation_space, flatten=True)
        value_h = h

        for n in hidden_units:
            h = layers.Dense(n, activation=activation)(h)

        if value_network == "copy":
            for n in hidden_units:
                value_h = layers.Dense(n, activation=activation)(value_h)
        else:
            value_h = h

        logits = layers.Dense(action_space.n)(h)
        values = layers.Dense(1)(value_h)
        values = layers.Reshape([])(values)

        if value_network is None:
            outputs = logits
        else:
            outputs = [logits, values]

        self._net = keras.Model(input_tensors, outputs)
        self._using_value_net = value_network is not None

    @property
    def trainable_weights(self) -> List[Tensor]:
        return self._net.trainable_weights

    def __call__(
        self, obs: Observation, training: bool = False
    ) -> Tuple[Distribution, Optional[Tensor]]:
        outputs = self._net(obs, training=training)

        if self._using_value_net:
            logits, values = outputs
        else:
            logits, values = outputs, None

        logits = maybe_mask_logits(
            logits, obs, self.observation_space, self.action_space
        )

        return Categorical(logits), values

    def get_weights(self) -> Weights:
        return self._net.get_weights()

    def set_weights(self, weights: Weights):
        self._net.set_weights(weights)
