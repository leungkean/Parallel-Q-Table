from typing import Dict, Any

import einops
import gym
import keras
import tensorflow as tf
from keras import layers

from afa.networks.models.base import Model
from afa.typing import Observation


def build_model(
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Discrete,
    model_config: Dict[str, Any],
) -> Model:
    if isinstance(observation_space, gym.spaces.Dict):
        obs_rank = len(observation_space["observed"].shape)
    else:
        obs_rank = len(observation_space.shape)

    if obs_rank == 1:
        from afa.networks.models.mlp import MLPModel

        return MLPModel(observation_space, action_space, **model_config)
    elif obs_rank == 3:
        from afa.networks.models.cnn import CNNModel

        return CNNModel(observation_space, action_space, **model_config)

    raise ValueError("No model for provided observation space.")


def create_input_tensors(observation_space: gym.Space, flatten: bool = False):
    if isinstance(observation_space, gym.spaces.Dict):
        input_tensors = {
            name: keras.Input(shape=space.shape, name=name)
            for name, space in observation_space.items()
        }

        to_concat = []

        if "observed" in set(observation_space.keys()) and "mask" in set(
            observation_space.keys()
        ):
            x_o = layers.Multiply()([input_tensors["observed"], input_tensors["mask"]])
            to_concat.append(x_o)
            to_concat.append(input_tensors["mask"])

        for name in sorted(input_tensors.keys()):
            if name not in {"observed", "mask"}:
                to_concat.append(input_tensors[name])

        if flatten:
            for i in range(len(to_concat)):
                to_concat[i] = layers.Flatten()(to_concat[i])

        h = layers.Concatenate()(to_concat)
    else:
        input_tensors = h = keras.Input(shape=observation_space.shape)

        if flatten:
            h = layers.Flatten()(h)

    return input_tensors, h


def maybe_mask_logits(
    logits: tf.Tensor,
    obs: Observation,
    observation_space: gym.Space,
    action_space: gym.spaces.Discrete,
) -> tf.Tensor:
    if isinstance(observation_space, gym.spaces.Dict) and "mask" in set(
        observation_space.keys()
    ):
        flat_mask = einops.rearrange(obs["mask"], "b ... -> b (...)")
        padding_amount = action_space.n - flat_mask.shape[-1]
        padded_mask = tf.pad(flat_mask, [(0, 0), (0, padding_amount)])
        logits = tf.where(padded_mask == 1, -1e12, logits)
        return logits

    return logits
