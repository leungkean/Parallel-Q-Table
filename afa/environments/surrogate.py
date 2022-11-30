import abc
from typing import Dict, Any

import gym
import numpy as np

from afa.environments.core import AcquisitionEnv
from afa.typing import Observation, ConfigDict


class SurrogateModel(abc.ABC):
    """An abstract base class for surrogate models.

    Surrogate models can be used in conjunction with the SurrogateWrapper to provide
    more meaningful rewards and enhanced observations to acquisition environments.

    Note that SurrogateModel subclasses must be able to be made into Ray Actors.
    This means that their constructor can only accept serializable config dictionaries.
    This also means that you should not directly decorate any of the abstract methods
    in the subclass (e.g. with @tf.function), as they will then not be visible for
    remote calls. Rather, you would need to decorate a helper function which is then
    called by the abstract method.

    Args:
        config: A serializable configuration dict.
    """

    @abc.abstractmethod
    def __init__(self, config: ConfigDict):
        pass

    @abc.abstractmethod
    def get_side_information_space(
        self, observation_space: gym.spaces.Dict
    ) -> gym.spaces.Box:
        """Returns the space of the side information provided by this model.

        Args:
            observation_space: The observation space of an acquisition environment.

        Returns:
            The space of the side information that will be provided by this model,
            given that it will be applied to an environment with `observation_space`
            as the original space.
        """
        pass

    @abc.abstractmethod
    def get_side_information(self, obs: Observation) -> np.ndarray:
        """Returns the side information for a given state.

        Args:
            obs: The observation from the environment for which to produce
                side information.

        Returns:
            The side information that will be added to the environment state.
        """
        pass

    @abc.abstractmethod
    def get_intermediate_reward(
        self, prev_obs: Observation, obs: Observation, info: Dict[str, Any]
    ) -> float:
        """Returns that intermediate reward that should be added at a given transition.

        Args:
            prev_obs: The observation before the action (acquisition) was taken.
            obs: The observation after the action (acquisition) was taken.
            info: The corresponding info dict from the environment, which will
                generally contain the true features (and true target) and may be useful
                in computing the intermediate reward.

        Returns:
            The intermediate reward.
        """
        pass

    @abc.abstractmethod
    def get_terminal_reward(self, obs: Observation, info: Dict[str, Any]) -> float:
        """Returns the terminal reward that should be added at the end of the episode.

        Args:
            obs: The final observation in the episode (i.e. the observation in which
                the terminal action was taken).
            info: The corresponding info dict from the environment, which will
                generally contain the true features (and true target) and may be useful
                in computing the final reward.

        Returns:
            The final reward.
        """
        pass

    def update_info(self, info: Dict[str, Any], done: bool):
        """Updates the current timestep's info dict.

        The info dict that is returned by this method is the one that will be returned
        from the environment. Note that this will always be called after either
        `get_intermediate_reward` or `get_terminal_reward`.

        Args:
            info: The info dict for the current timestep.
            done: Whether or not the current timestep is terminal.

        Returns:
            A possibly modified info dict for the current timestep.
        """
        return info


class SurrogateWrapper(gym.Wrapper):
    """Environment wrapper that applies the rewards and side info from a SurrogateModel.

    Args:
        env: The AcquisitionEnv to be wrapped.
        model: The SurrogateModel that will be used to generate side info and rewards.
    """

    def __init__(self, env: gym.Env, model: SurrogateModel):
        assert isinstance(env, AcquisitionEnv)
        super().__init__(env)
        self.model = model

        self.observation_space = gym.spaces.Dict(
            {
                **env.observation_space.spaces,
                "side_info": model.get_side_information_space(env.observation_space),
            }
        )

    def _augment_obs(self, obs):
        side_info = self.model.get_side_information(obs)
        obs["side_info"] = np.asarray(side_info, dtype=np.float32)
        return obs

    def step(self, action):
        prev_obs = self.env._get_observation()
        obs, reward, done, info = self.env.step(action)

        if not done:
            reward += np.asarray(
                self.model.get_intermediate_reward(prev_obs, obs, info)
            ).item()
        else:
            reward += np.asarray(self.model.get_terminal_reward(obs, info)).item()

        info = self.model.update_info(info, done)

        return self._augment_obs(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._augment_obs(obs)
