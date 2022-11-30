from typing import Dict, Any, Optional

import gym
import numpy as np
import ray

from afa.environments.surrogate import SurrogateModel
from afa.surrogate.ace import ACEInstanceRecognitionSurrogateModel
from afa.surrogate.deformer.surrogate import (
    DEformerClassificationSurrogateModel,
    DEformerInstanceRecognitionSurrogateModel,
)
from afa.typing import Observation, ConfigDict


class RemoteSurrogateModel(SurrogateModel):
    """Wrapper class around a remote surrogate model."""

    def __init__(self, surrogate_actor: ray.actor.ActorHandle):
        self._model = surrogate_actor

    def get_side_information_space(
        self, observation_space: gym.spaces.Dict
    ) -> gym.spaces.Box:
        return ray.get(self._model.get_side_information_space.remote(observation_space))

    def get_side_information(self, obs: Observation) -> np.ndarray:
        return ray.get(self._model.get_side_information.remote(obs))

    def get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        return ray.get(self._model.get_intermediate_reward.remote(prev_obs, obs, info))

    def get_terminal_reward(self, obs: Observation, info: Dict[str, Any]) -> float:
        return ray.get(self._model.get_terminal_reward.remote(obs, info))

    def update_info(self, info: Dict[str, Any], done: bool):
        return ray.get(self._model.update_info.remote(info, done))


def get_surrogate_model(
    surrogate_type: str,
    surrogate_config: ConfigDict,
    run_remote: bool = False,
    num_cpus: Optional[int] = 1,
    num_gpus: float = 0,
) -> SurrogateModel:
    """Creates a specified surrogate model.

    All surrogate models should be created through this method. Note that when
    new surrogate model subclasses are created, this method should be updated
    accordingly to make them accessible.

    Args:
        surrogate_type: The type of surrogate model to create. This string will
            typically be derived from the type of the surrogate artifact being used
            and the type of environment (e.g. air).
        surrogate_config: The config dict for the surrogate model.
        run_remote: If True, the surrogate model will be run in its own process.
        num_cpus: The number of CPUs to allocated to each surrogate model process if
            `run_remote=True`.
        num_gpus: The number of GPUs to allocated to each surrogate model process if
            `run_remote=True`.

    Returns:
        A `SurrogateModel`.
    """
    if surrogate_type == "air_ace_model":
        model_cls = ACEInstanceRecognitionSurrogateModel
    elif surrogate_type == "classification_deformer_model":
        model_cls = DEformerClassificationSurrogateModel
    elif surrogate_type == "air_deformer_model":
        model_cls = DEformerInstanceRecognitionSurrogateModel
    else:
        raise ValueError(f"{surrogate_type} is not a valid surrogate model type.")

    if run_remote:
        actor = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(model_cls).remote(
            surrogate_config
        )
        model = RemoteSurrogateModel(actor)
    else:
        model = model_cls(surrogate_config)

    return model
