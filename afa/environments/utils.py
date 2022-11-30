from typing import Callable, Tuple, Union

import gym

from afa.data import load_unsupervised_split_as_numpy, load_supervised_split_as_numpy
from afa.environments.core import (
    InstanceRecognitionEnv,
    IndirectClassificationEnv,
    PretrainedClassifierEnv,
)
from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.environments.surrogate import SurrogateWrapper
from afa.networks.classifiers.utils import get_environment_classifier_fn
from afa.rewards import get_reward_fn
from afa.surrogate.utils import get_surrogate_model
from afa.typing import ConfigDict


def create_surrogate_air_env_fn(
    dataset: str,
    data_split: str,
    error_on_new_epoch: bool = False,
    return_dataset_manager: bool = False,
) -> Union[
    Callable[[ConfigDict], gym.Env],
    Tuple[Callable[[ConfigDict], gym.Env], EnvironmentDatasetManager],
]:
    """Creates an instance recognition env that is wrapped with a surrogate model.

    Args:
        dataset: The dataset to generate instances from.
        data_split: The dataset split to use.
        error_on_new_epoch: If True, then an error will be raised if more than one
            epoch's worth of episodes are attempted.
        return_dataset_manager: If True, then the environments' dataset manager
            will also be returned.

    Returns:
        A function that creates the environment.
    """
    features = load_unsupervised_split_as_numpy(dataset, data_split)
    dataset_manager = EnvironmentDatasetManager.remote(
        features, error_on_new_epoch=error_on_new_epoch
    )

    def env_fn(config: ConfigDict) -> gym.Env:
        surrogate_model = get_surrogate_model(
            config["surrogate"]["type"],
            config["surrogate"]["config"],
            config["surrogate"].get("run_remote", False),
            config["surrogate"].get("num_cpus", None),
            config["surrogate"].get("num_gpus", 0),
        )

        env = InstanceRecognitionEnv(
            dataset_manager, acquisition_cost=config["acquisition_cost"]
        )
        env = SurrogateWrapper(env, surrogate_model)
        return env

    if return_dataset_manager:
        return env_fn, dataset_manager

    return env_fn


def create_surrogate_classification_env_fn(
    dataset: str,
    data_split: str,
    error_on_new_epoch: bool = False,
    return_dataset_manager: bool = False,
) -> Union[
    Callable[[ConfigDict], gym.Env],
    Tuple[Callable[[ConfigDict], gym.Env], EnvironmentDatasetManager],
]:
    """Creates a classification env that is wrapped with a surrogate model.

    Args:
        dataset: The dataset to generate instances from.
        data_split: The dataset split to use.
        error_on_new_epoch: If True, then an error will be raised if more than one
            epoch's worth of episodes are attempted.
        return_dataset_manager: If True, then the environments' dataset manager
            will also be returned.

    Returns:
        A function that creates the environment.
    """
    features, targets = load_supervised_split_as_numpy(dataset, data_split)
    dataset_manager = EnvironmentDatasetManager.remote(
        features, targets, error_on_new_epoch=error_on_new_epoch
    )

    def env_fn(config: ConfigDict) -> gym.Env:
        surrogate_model = get_surrogate_model(
            config["surrogate"]["type"],
            config["surrogate"]["config"],
            config["surrogate"].get("run_remote", False),
            config["surrogate"].get("num_cpus", None),
            config["surrogate"].get("num_gpus", 0),
        )

        env = IndirectClassificationEnv(
            dataset_manager, acquisition_cost=config["acquisition_cost"]
        )
        env = SurrogateWrapper(env, surrogate_model)
        return env

    if return_dataset_manager:
        return env_fn, dataset_manager

    return env_fn


def create_pretrained_classifier_env_fn(
    dataset: str,
    data_split: str,
    error_on_new_epoch: bool = False,
    return_dataset_manager: bool = False,
) -> Union[
    Callable[[ConfigDict], gym.Env],
    Tuple[Callable[[ConfigDict], gym.Env], EnvironmentDatasetManager],
]:
    """Creates a classification env that defers to a pretrained classifier at terminal steps.

    Args:
        dataset: The dataset to generate instances from.
        data_split: The dataset split to use.
        error_on_new_epoch: If True, then an error will be raised if more than one
            epoch's worth of episodes are attempted.
        return_dataset_manager: If True, then the environments' dataset manager
            will also be returned.

    Returns:
        A function that creates the environment.
    """
    features, targets = load_supervised_split_as_numpy(dataset, data_split)
    dataset_manager = EnvironmentDatasetManager.remote(
        features, targets, error_on_new_epoch=error_on_new_epoch
    )

    def env_fn(config: ConfigDict) -> gym.Env:
        env_classifier_fn = get_environment_classifier_fn(config["classifier_dir"])
        terminal_reward_fn = get_reward_fn(
            config["reward_type"], **config.get("reward_kwargs", {})
        )

        env = PretrainedClassifierEnv(
            dataset_manager,
            env_classifier_fn,
            terminal_reward_fn=terminal_reward_fn,
            acquisition_cost=config["acquisition_cost"],
        )
        return env

    if return_dataset_manager:
        return env_fn, dataset_manager

    return env_fn
