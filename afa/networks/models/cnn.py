from typing import Optional, Sequence, Tuple, List

import gym
import keras
from keras import layers
from tensorflow import Tensor
from tensorflow_probability.python.distributions import Categorical
from tensorflow_probability.python.distributions import Distribution

from afa.networks.models.base import Model
from afa.networks.models.utils import create_input_tensors, maybe_mask_logits
from afa.typing import Weights, Observation


class CNNModel(Model):
    """A simple convolutional model.

    A value network can optionally be included as either a copy of the policy network
    or a separate head that shares a body with the policy network.

    Args:
        observation_space: The environment's observation space.
        action_space: The environment's action space.
        conv_layers: A list of tuples where the ith tuple specifies the
            (filters, kernel, stride) in the ith convolutional layer.
        hidden_units: A sequence of integers, where the ith value is the number of
            units in the ith hidden layer after the convolutions.
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
        conv_layers: Sequence[Tuple[int, int, int]] = (
            (16, 3, 2),
            (32, 3, 2),
        ),
        hidden_units: Sequence[int] = (32, 32),
        activation: str = "leaky_relu",
        value_network: Optional[str] = None,
    ):
        super().__init__(observation_space, action_space)

        assert value_network in {None, "copy", "shared"}

        input_tensors, h = create_input_tensors(observation_space)
        value_h = h

        for filters, kernel, stride in conv_layers:
            h = layers.Conv2D(
                filters, kernel, stride, padding="SAME", activation=activation
            )(h)

        h = layers.Flatten()(h)

        for n in hidden_units:
            h = layers.Dense(n, activation=activation)(h)

        if value_network == "copy":
            for filters, kernel, stride in conv_layers:
                value_h = layers.Conv2D(
                    filters, kernel, stride, padding="SAME", activation=activation
                )(value_h)

            value_h = layers.Flatten()(value_h)

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
