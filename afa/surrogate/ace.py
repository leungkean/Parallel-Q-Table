import json
import os
from typing import Dict, Any

import gym
import numpy as np
import tensorflow as tf
from ace import ACEModel

from afa.environments.surrogate import SurrogateModel
from afa.typing import Observation, Array, ConfigDict


class ACEInstanceRecognitionSurrogateModel(SurrogateModel):
    """A surrogate model for instance recognition that is backed by ACE."""

    def __init__(self, config: ConfigDict):
        with open(os.path.join(config["model_dir"], "model_config.json"), "r") as fp:
            model_config = json.load(fp)

        self._model = ACEModel(**model_config)
        self._model.load_weights(os.path.join(config["model_dir"], "weights.h5"))

        self._side_info_num_samples = config.get("side_info_num_samples", 10)

    def get_side_information_space(
        self, observation_space: gym.spaces.Dict
    ) -> gym.spaces.Box:
        data_shape = observation_space["observed"].shape
        return gym.spaces.Box(
            -np.inf, np.inf, shape=(*data_shape[:-1], 2 * data_shape[-1])
        )

    @tf.function
    def _get_side_information(self, obs: Observation) -> np.ndarray:
        print("Tracing ACEInstanceRecognitionSurrogateModel.get_side_information...")

        x_o = tf.expand_dims(obs["observed"], 0)
        b = tf.expand_dims(obs["mask"], 0)
        samples = self._model.sample(
            x_o, b, num_samples=self._side_info_num_samples, use_proposal=True
        )
        samples = tf.squeeze(samples, 0)
        samples = tf.where(b == 1, x_o, samples)
        mean, var = tf.nn.moments(samples, axes=[0])
        return tf.concat([mean, var], axis=-1)

    @tf.function
    def _get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        print("Tracing ACEInstanceRecognitionSurrogateModel.get_intermediate_reward...")

        x_o = tf.stack([prev_obs["observed"], obs["observed"]], axis=0)
        b = tf.stack([prev_obs["mask"], obs["mask"]], axis=0)

        proposal_dist, _ = self._model._proposal_network([x_o, b], training=False)
        lls = proposal_dist.log_prob(tf.expand_dims(info["truth"], 0))
        lls = tf.reduce_sum(lls * (1 - b), axis=-1)

        bpds = tf.math.divide_no_nan(
            -lls, tf.reduce_sum(1 - b, axis=1) + 1e-8
        ) / tf.math.log(2.0)

        return bpds[0] - bpds[1]

    @tf.function
    def _sse(self, obs: Observation, truth: Array) -> float:
        print("Tracing ACEInstanceRecognitionSurrogateModel._sse...")

        proposal_dist, _ = self._model._proposal_network(
            [tf.expand_dims(obs["observed"], 0), tf.expand_dims(obs["mask"], 0)],
            training=False,
        )
        imputations = tf.squeeze(proposal_dist.mean(), 0)
        sse = tf.reduce_sum((truth - imputations) ** 2 * (1 - obs["mask"]))

        return sse

    def get_side_information(self, obs: Observation) -> np.ndarray:
        return self._get_side_information(obs)

    def get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        return self._get_intermediate_reward(prev_obs, obs, info)

    def get_terminal_reward(self, obs: Observation, info: Dict[str, Any]) -> float:
        self._last_sse = self._sse(obs, info["truth"])
        return -self._last_sse

    def update_info(self, info: Dict[str, Any], done: bool):
        if done:
            info.setdefault("episode", {})["sse"] = np.asarray(self._last_sse).item()

        return info
