import json
import os
import pickle
import random
from typing import Dict, Any

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from bax.utils import set_jax_memory_preallocation

from afa.environments.surrogate import SurrogateModel
from afa.surrogate.deformer.models import ClassificationDEformer, ContinuousDEformer
from afa.typing import Observation, ConfigDict


class DEformerClassificationSurrogateModel(SurrogateModel):
    """A surrogate model for classification that is backed by DEformer."""

    def __init__(self, config: ConfigDict):
        set_jax_memory_preallocation(False)

        with open(os.path.join(config["model_dir"], "model_config.json"), "r") as fp:
            model_config = json.load(fp)

        with open(os.path.join(config["model_dir"], "state.pkl"), "rb") as fp:
            params = pickle.load(fp).params

        self._prng = hk.PRNGSequence(random.randrange(int(2e9)))

        def predict_fn(obs):
            model = ClassificationDEformer(**model_config)
            logits = model.predict(
                jnp.expand_dims(obs["observed"], axis=0),
                jnp.expand_dims(obs["mask"], axis=0),
            )
            return jnp.squeeze(logits, axis=0)

        def side_info_fn(obs):
            model = ClassificationDEformer(**model_config)

            info_gains = model.expected_info_gains(
                obs["observed"],
                obs["mask"],
                num_samples=config.get("num_samples", 10),
                evaluation_method=config.get(
                    "info_gains_evaluation_method", "vectorized"
                ),
            )

            return info_gains

        def intermediate_reward_fn(prev_obs, obs):
            x_o = jnp.stack([prev_obs["observed"], obs["observed"]], axis=0)
            b = jnp.stack([prev_obs["mask"], obs["mask"]], axis=0)

            model = ClassificationDEformer(**model_config)
            logits = model.predict(x_o, b)

            entropy = -jnp.sum(jnp.exp(logits) * logits, axis=-1)

            return entropy[0] - entropy[1]

        def terminal_reward_fn(obs, target):
            logits = predict_fn(obs)
            target = jax.nn.one_hot(target, logits.shape[0])
            reward = jnp.sum(target * logits)
            return reward, logits

        side_info_fn = jax.jit(hk.transform(side_info_fn).apply)
        terminal_reward_fn = jax.jit(hk.transform(terminal_reward_fn).apply)
        intermediate_reward_fn = jax.jit(hk.transform(intermediate_reward_fn).apply)

        self._side_info = lambda obs: np.asarray(
            side_info_fn(params, self._prng.next(), obs)
        )

        self._terminal_reward = lambda obs, target: terminal_reward_fn(
            params, self._prng.next(), obs, target
        )

        self._intermediate_reward = lambda prev_obs, obs: intermediate_reward_fn(
            params, self._prng.next(), prev_obs, obs
        )

    def get_side_information_space(
        self, observation_space: gym.spaces.Dict
    ) -> gym.spaces.Box:
        return gym.spaces.Box(-np.inf, np.inf, shape=observation_space["mask"].shape)

    def get_side_information(self, obs: Observation) -> np.ndarray:
        return self._side_info(obs)

    def get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        return self._intermediate_reward(prev_obs, obs)

    def get_terminal_reward(self, obs: Observation, info: Dict[str, Any]) -> float:
        reward, self._last_logits = self._terminal_reward(obs, info["target"])
        return reward

    def update_info(self, info: Dict[str, Any], done: bool):
        if done:
            info.setdefault("episode", {})["accuracy"] = float(
                np.argmax(self._last_logits) == info["target"]
            )
            info["classifier_logits"] = np.asarray(self._last_logits)

        return info


class DEformerInstanceRecognitionSurrogateModel(SurrogateModel):
    """A surrogate model for instance recognition that is backed by DEformer."""

    def __init__(self, config: ConfigDict):
        set_jax_memory_preallocation(False)

        with open(os.path.join(config["model_dir"], "model_config.json"), "r") as fp:
            model_config = json.load(fp)

        with open(os.path.join(config["model_dir"], "state.pkl"), "rb") as fp:
            params = pickle.load(fp).params

        self._prng = hk.PRNGSequence(random.randrange(int(2e9)))

        def side_info_fn(obs):
            model = ContinuousDEformer(**model_config)

            x = jnp.expand_dims(obs["observed"], axis=0)
            b = jnp.expand_dims(obs["mask"], axis=0)

            noise = jax.random.uniform(hk.next_rng_key(), x.shape) - b
            order = jnp.argsort(noise, axis=-1)

            dist = model.get_conditional_distributions(x, b, order)

            mean = dist.mean() * (1 - b)
            var = dist.variance() * (1 - b)

            side_info = jnp.concatenate([mean, var], axis=-1)
            side_info = jnp.squeeze(side_info, 0)

            return side_info

        def intermediate_reward_fn(prev_obs, obs, info):
            x_o = jnp.stack([prev_obs["observed"], obs["observed"]], axis=0)
            b = jnp.stack([prev_obs["mask"], obs["mask"]], axis=0)

            model = ContinuousDEformer(**model_config)

            noise = jax.random.uniform(hk.next_rng_key(), x_o.shape) - b
            order = jnp.argsort(noise, axis=-1)

            dist = model.get_conditional_distributions(x_o, b, order)

            lls = dist.log_prob(jnp.expand_dims(info["truth"], 0))
            lls = jnp.sum(lls * (1 - b), axis=-1)

            u = jnp.sum(1 - b, axis=1)
            bpds = jnp.nan_to_num(-lls / (u + 1e-8)) / jnp.log(2)

            return bpds[0] - bpds[1]

        def sse_fn(obs, truth):
            model = ContinuousDEformer(**model_config)

            x = jnp.expand_dims(obs["observed"], axis=0)
            b = jnp.expand_dims(obs["mask"], axis=0)

            noise = jax.random.uniform(hk.next_rng_key(), x.shape) - b
            order = jnp.argsort(noise, axis=-1)

            imputations = model.impute(x, b, order)
            imputations = jnp.squeeze(imputations, 0)

            sse = jnp.sum((truth - imputations) ** 2 * (1 - obs["mask"]))

            return sse

        side_info_fn = jax.jit(hk.transform(side_info_fn).apply)
        sse_fn = jax.jit(hk.transform(sse_fn).apply)
        intermediate_reward_fn = jax.jit(hk.transform(intermediate_reward_fn).apply)

        self._side_info = lambda obs: np.asarray(
            side_info_fn(params, self._prng.next(), obs)
        )

        self._sse = lambda obs, truth: sse_fn(params, self._prng.next(), obs, truth)

        self._intermediate_reward = lambda prev_obs, obs, info: intermediate_reward_fn(
            params, self._prng.next(), prev_obs, obs, info
        )

    def get_side_information_space(
        self, observation_space: gym.spaces.Dict
    ) -> gym.spaces.Box:
        data_shape = observation_space["observed"].shape
        return gym.spaces.Box(
            -np.inf, np.inf, shape=(*data_shape[:-1], 2 * data_shape[-1])
        )

    def get_side_information(self, obs: Observation) -> np.ndarray:
        return self._side_info(obs)

    def get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        return self._intermediate_reward(prev_obs, obs, info)

    def get_terminal_reward(self, obs: Observation, info: Dict[str, Any]) -> float:
        self._last_sse = self._sse(obs, info["truth"])
        return -self._last_sse

    def update_info(self, info: Dict[str, Any], done: bool):
        if done:
            info.setdefault("episode", {})["sse"] = np.asarray(self._last_sse).item()

        return info
