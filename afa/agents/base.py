import abc
import os
import pickle
from collections import deque
from typing import Callable, Dict, Any, Tuple, List, Optional, Sequence

import gym
import numpy as np
from ray.rllib import SampleBatch
from ray.util.ml_utils.dict import deep_update

from afa.policies.base import Policy
from afa.typing import ConfigDict, Numeric


class Callback(abc.ABC):
    """Base class for agent training callbacks."""

    def on_train_result(self, step: int, logs: Dict[str, Any]):
        """This method is called after each training step.

        Args:
            step: The training iteration.
            logs: A dictionary containing metrics from the training step.

        Returns:
            None.
        """
        pass


class WandbCallback(Callback):
    """Callback that logs training results to W&B.

    Args:
        run: The wandb run to log metrics to.
    """

    def __init__(self, run: "wandb.sdk.wandb_run.Run"):
        self._run = run

    def on_train_result(self, step: int, logs: Dict[str, Any]):
        self._run.log(logs, step=step)


class Agent(abc.ABC):
    """A base class for agents.

    Args:
        config: A configuration dictionary for the agent.
        env_fn: A function that accepts a configuration dictionary and returns an
            environment.
    """

    def __init__(self, config: ConfigDict, env_fn: Callable[[ConfigDict], gym.Env]):
        self._config = self.validate_config(config)
        self._env_fn = env_fn

        env = env_fn(config.get("env_config", {}))
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    @property
    def config(self) -> ConfigDict:
        """The agent's config."""
        return self._config

    @property
    def env_fn(self) -> Callable[[ConfigDict], gym.Env]:
        """The agent's environment creation function"""
        return self._env_fn

    @property
    @abc.abstractmethod
    def default_config(self) -> ConfigDict:
        """The default config for agent."""
        pass

    @property
    @abc.abstractmethod
    def policy(self) -> Policy:
        """The agent's policy."""
        pass

    @abc.abstractmethod
    def train_setup(self):
        """Prepares the agent for training.

        This function is called once before training starts.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def train_step(self, step: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes one training iteration.

        Args:
            step: The current training iteration.

        Returns:
            metrics: A dictionary containing metrics about the training step.
            stats: A dictionary containing stats produced by the rollout workers.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, num_episodes: int) -> Tuple[Dict[str, Any], List[SampleBatch]]:
        """Runs the agent in evaluation mode for some number of episodes.

        Args:
            num_episodes: The number of episodes to complete.

        Returns:
            A list of dictionaries with statistics about the episodes and a list of
            the episode `SampleBatch`s.
        """
        pass

    def validate_config(self, config: ConfigDict) -> ConfigDict:
        """Ensures that the provided config is valid.

        This function is called when an `Agent` is initialized. This function can be
        overridden to do custom validation. By default, it just merges the provided
        config with the agent's defaulut config.

        Args:
            config: The config dict to validate.

        Returns:
            The validated config dict.
        """
        merged_config = deep_update(self.default_config, config, new_keys_allowed=True)

        if "env_config" not in merged_config:
            merged_config["env_config"] = {}

        if "evaluation_config" not in merged_config:
            merged_config["evaluation_config"] = {}

        return merged_config

    def train(
        self,
        num_iterations: int,
        save_dir: str,
        episode_window_size: int = 100,
        num_eval_episodes: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Trains the agent.

        This method will train the agent by repeatedly calling `train_step`. It will
        also handle some metric tracking and weight saving.

        Args:
            num_iterations: The number of iterations to train for.
            save_dir: The directory in which training data will be saved.
            episode_window_size: The window size over which episode statistics will
                be averaged.
            num_eval_episodes: The number of episodes to complete each time the agent
                is evaluated. If None, then evaluation will not run.
            callbacks: A list of callbacks to run during training.

        Returns:
            None
        """
        self.train_setup()

        ep_info_buffer = deque([], maxlen=episode_window_size)
        callbacks = callbacks or []
        best_reward_mean = -np.inf
        total_episodes = 0
        total_timesteps = 0

        os.makedirs(save_dir, exist_ok=True)

        for step in range(num_iterations):
            logs, stats = self.train_step(step)
            ep_info_buffer.extend(stats["episode_infos"])

            total_episodes += stats["num_episodes"]
            total_timesteps += stats["num_timesteps"]

            logs["total_timesteps"] = total_timesteps
            logs["total_episodes"] = total_episodes
            logs["mean_policy_eval_ms"] = stats["mean_policy_eval_ms"]

            current_reward = None

            if ep_info_buffer:
                episode_data = _parse_episode_infos(ep_info_buffer)

                logs.update(episode_data)

                print_string = (
                    f"[Step {step}] - mean_episode_reward: "
                    f"{episode_data['mean_episode_reward']:.3f}"
                    f" - mean_episode_length: "
                    f"{episode_data['mean_episode_length']:.3f}"
                )
                print(print_string)

                current_reward = episode_data["mean_episode_reward"]

            if num_eval_episodes is not None:
                eval_stats, _ = self.evaluate(num_eval_episodes)
                eval_episode_data = _parse_episode_infos(eval_stats["episode_infos"])

                logs["val_mean_policy_eval_ms"] = eval_stats["mean_policy_eval_ms"]

                for k, v in eval_episode_data.items():
                    logs["val_" + k] = v

                current_reward = eval_episode_data["mean_episode_reward"]

            for callback in callbacks:
                callback.on_train_result(step, logs)

            if current_reward is not None and current_reward > best_reward_mean:
                self.save(os.path.join(save_dir, "best_weights.pkl"))

    def save(self, path: str):
        """Saves the agent's weights as a pickle.

        Args:
            path: The file to save the agent's weights to.

        Returns:
            None.
        """
        weights = self.policy.get_weights()

        with open(path, "wb") as fp:
            pickle.dump(weights, fp)

    def load(self, path: str):
        """Loads weights from disk into the agent.

        Args:
            path: Path to the file containing the weights to load.

        Returns:

        """
        with open(path, "rb") as fp:
            weights = pickle.load(fp)

        self.policy.set_weights(weights)


def _parse_episode_infos(
    ep_info_buffer: Sequence[Dict[str, Any]]
) -> Dict[str, Numeric]:
    episode_data = {}

    aggregators = {"min": np.min, "mean": np.mean, "max": np.max}

    for key in ep_info_buffer[-1].keys():
        values = [ep_info[key] for ep_info in ep_info_buffer]

        for n, fn in aggregators.items():
            episode_data[n + "_episode_" + key] = fn(values)

    return episode_data
