import abc
from typing import Optional, Union, Callable

import gym
import numpy as np

from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.typing import Observation


class AcquisitionEnv(gym.Env, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition environments.

    This class implements the functionality for an environment in which each episode
    consists of the agent sequential acquiring individual features of a single example
    in a dataset. At each timestep, the environment returns the features that have been
    observed so far in the episode and a binary mask indicating which features have
    been observed. A randomly selected instance in the dataset is chosen for each
    episode, and every example is seen one before repeating the dataset.

    Acquiring a feature is assumed to have an associated cost, and the reward produced
    by the environment after acquiring a given feature is the negative of the feature's
    acquisition cost. The costs can be uniform across all features, or they can vary.

    Choosing to reacquire a feature which has already been acquired will not change the
    environment state, but the acquisition cost of that feature will be incurred again.
    Generally, agents operating in this environment should implement a mechanism to
    prevent reacquiring features.

    While the action space is not specified for this abstract class, it is assumed that
    subclasses will have at least d + 1 actions, where d is the number of features.
    Actions 0, ..., d - 1 correspond to acquiring each of the d features, and actions
    larger than d - 1 are terminal.

    Args:
        dataset_manager: The dataset manager from which examples will be sourced.
        index_dims: This is the number of leftmost dimensions of a single example that
            should be interpreted as the feature index. The remaining dimensions are
            considered to be channels that belong to a single feature. For example, if
            working with images of the shape [64, 64, 3] and you want to acquire entire
            pixels at a time (rather than individually acquiring RGB values within a
            pixel), then `index_dims` should be set to 2. This also implies that the
            binary masks included in the environment observations will have shape
            [64, 64], not [64, 64, 3]. By default, all dimensions are assumed to be
            index dimensions.
        acquisition_cost: If this is a scalar, then every feature will have the same
            acquisition cost. This can also be an array with the same shape as the
            data's index dimensions, in which case each value corresponds to the cost
            of acquiring the feature at that location.
        max_acquisitions: The maximum number of acquisitions that are allowed in each
            episode. If the limit is reached, then the episode is forced to terminate.
            By default, there is no limit.
    """

    def __init__(
        self,
        dataset_manager: EnvironmentDatasetManager,
        index_dims: Optional[int] = None,
        acquisition_cost: Optional[Union[np.ndarray, float]] = None,
        max_acquisitions: Optional[int] = None,
    ):
        self._dataset_manager = dataset_manager

        data_shape = self._dataset_manager.features_shape()
        index_dims = index_dims or len(data_shape)
        assert index_dims <= len(data_shape)
        self._index_dims = index_dims
        self._index_shape = data_shape[: 1 + index_dims]
        self._target_shape = self._dataset_manager.targets_shape()

        if not self._target_shape:
            self._target_shape = (1,)

        self._current_features = None
        self.current_observed_mask = np.zeros(self._index_shape, dtype=np.bool_)

        #acquisition_cost = acquisition_cost or 0.0
        if isinstance(acquisition_cost, float):
            self.acquisition_cost = np.ones(self._index_shape) * acquisition_cost
        else:
            assert acquisition_cost.shape == self._index_shape
            self.acquisition_cost = acquisition_cost

        self.max_acquisitions = max_acquisitions
        self.terminal = 0

        self.observation_space = gym.spaces.Dict(
            {
                "observed": gym.spaces.Box(-np.inf, np.inf, data_shape),
                "mask": gym.spaces.Box(0, 1, self._index_shape),
            }
        )

    @property
    def num_features(self):
        return np.prod(self._index_shape)

    @property
    def current_example(self):
        return self._current_features

    def _get_observation(self):
        observed = self.current_example * self.current_observed_mask
        return {
            "observed": observed,
            "mask": self.current_observed_mask.astype(observed.dtype),
        }

    def _compute_reward(self, action):
        if action < self.num_features:
            return -np.reshape(self.acquisition_cost, [-1])[action]

        return 0.0

    def reset(self):
        self._current_features = self._dataset_manager.get_new_instance()

        self.current_observed_mask[:] = False

        return self._get_observation()

    def step(self, action):
        reward = self._compute_reward(action)

        if action < self.num_features:
            # Update a flattened view of the mask
            np.reshape(self.current_observed_mask, [-1])[action] = True

        self.terminal += int(action >= self.num_features)
        done = self.terminal == self._target_shape[0]
        if done:
            self.terminal = 0

        if (
            self.max_acquisitions is not None
            and np.count_nonzero(self.current_observed_mask) >= self.max_acquisitions
            and self.terminal == self._target_shape[0]
        ):
            done = True

        info = {"truth": self.current_example.copy()}

        return self._get_observation(), reward, done, info


class DirectClassificationEnv(AcquisitionEnv):
    """An acquisition environment where the goal is classification.

    In this version, the agent has n*t terminal actions available to it, one for each
    of the n possible classes for each target label. The agent is directly trying to predict the class
    in this environment.

    Args:
        dataset_manager: The dataset manager from which features and targets will be
            sourced. It is assumed that this manager contains target values, and that
            those values range from 0 to n - 1, where n is the number of classes.
        correct_reward: The reward received for a correct classification.
        incorrect_reward: The reward received for an incorrect classification.
    """

    def __init__(
        self,
        dataset_manager: EnvironmentDatasetManager,
        correct_reward: Optional[float] = None,
        incorrect_reward: Optional[float] = None,
        **kwargs
    ):
        super().__init__(dataset_manager, **kwargs)

        self._current_target = None
        self._num_classes = self._dataset_manager.num_classes()
        self._num_targets = self._dataset_manager.num_targets()

        self.correct_reward = correct_reward or 0.0
        self.incorrect_reward = incorrect_reward or 0.0

        self.action_space = gym.spaces.Discrete(self.num_features + self.num_classes * self.num_targets)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_targets(self):
        return self._num_targets

    @property
    def current_target(self):
        return self._current_target

    def reset(self):
        self._current_features, self._current_target = self._dataset_manager.get_new_instance()

        self.current_observed_mask[:] = False

        return self._get_observation()

    def _compute_reward(self, action):
        reward = super()._compute_reward(action)

        if action >= self.num_features and self.current_target.shape == ():
            correct_terminal = self.current_target
            correct = action - self.num_features == correct_terminal
            reward += self.correct_reward if correct else self.incorrect_reward
        elif action >= self.num_features:
            correct_terminal = self.current_target[(action - self.num_features)//self.num_classes] + ((action - self.num_features)//self.num_classes) * self.num_classes
            correct_terminal += self.num_features
            correct = action == correct_terminal
            reward += self.correct_reward if correct else self.incorrect_reward

        return reward

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["target"] = self.current_target
        return obs, reward, done, info


class IndirectClassificationEnv(AcquisitionEnv):
    """An acquisition environment where the goal is classification.

    In this version, the agent has one terminal action available. However, there is
    no classifier connected to this class in any way. It is generally assumed that
    this class will be wrapped with e.g. a surrogate model that makes classification
    decisions or that class predictions are otherwise externally produced.

    Args:
        dataset_manager: The dataset manager from which features and targets will be
            sourced. It is assumed that this manager contains target values, and that
            those values range from 0 to n - 1, where n is the number of classes.
        correct_reward: The reward received for a correct classification.
        incorrect_reward: The reward received for an incorrect classification.
    """

    def __init__(
        self,
        dataset_manager: EnvironmentDatasetManager,
        correct_reward: Optional[float] = None,
        incorrect_reward: Optional[float] = None,
        **kwargs
    ):
        super().__init__(dataset_manager, **kwargs)

        self.correct_reward = correct_reward or 0.0
        self.incorrect_reward = incorrect_reward or 0.0

        self.action_space = gym.spaces.Discrete(self.num_features + 1)

        self._current_target = None
        self._num_classes = self._dataset_manager.num_classes()

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def current_target(self):
        return self._current_target

    def reset(self):
        self._current_features, self._current_target = self._dataset_manager.get_new_instance()

        self.current_observed_mask[:] = False

        return self._get_observation()

    def _compute_reward(self, action):
        reward = super()._compute_reward(action)

        if action >= self.num_features:
            correct = (action - self.num_features) == self.current_target
            reward += self.correct_reward if correct else self.incorrect_reward

        return reward

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["target"] = self.current_target
        return obs, reward, done, info


class PretrainedClassifierEnv(AcquisitionEnv):
    """An acquisition environment where the agent defers to a pretrained classifier.

    In this version, the agent has one terminal action available. When the terminal
    action is selected, the classification decision is made by a provided pretrained
    classifier, and the reward is based on that classifier's decision.

    Args:
        dataset_manager: The dataset manager from which features and targets will be
            sourced. It is assumed that this manager contains target values, and that
            those values range from 0 to n - 1, where n is the number of classes.
        classifier_fn: A function that accepts an environment observation and returns
            the predicted class.
        correct_reward: The reward received for a correct classification.
        incorrect_reward: The reward received for an incorrect classification.
        terminal_reward_fn: A function that can be used to optionally specify an
            arbitrary method for computing the terminal reward. This function should
            accept the true target and the classifier's logits, then return the
            terminal reward.
    """

    def __init__(
        self,
        dataset_manager: EnvironmentDatasetManager,
        classifier_fn: Callable[[Observation], np.ndarray],
        correct_reward: Optional[float] = None,
        incorrect_reward: Optional[float] = None,
        terminal_reward_fn: Optional[Callable[[int, np.ndarray], float]] = None,
        **kwargs
    ):
        super().__init__(dataset_manager, **kwargs)

        self.classifier_fn = classifier_fn
        self.correct_reward = correct_reward or 0.0
        self.incorrect_reward = incorrect_reward or 0.0
        self.terminal_reward_fn = terminal_reward_fn

        self.action_space = gym.spaces.Discrete(self.num_features + 1)

        self._last_logits = None
        self._current_target = None
        self._num_classes = self._dataset_manager.num_classes()

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def current_target(self):
        return self._current_target

    def reset(self):
        self._current_features, self._current_target = self._dataset_manager.get_new_instance()

        self.current_observed_mask[:] = False

        return self._get_observation()

    def _compute_reward(self, action):
        reward = super()._compute_reward(action)

        if action == self.num_features:
            self._last_logits = self.classifier_fn(self._get_observation())
            if self.terminal_reward_fn is not None:
                reward = np.asarray(
                    self.terminal_reward_fn(self.current_target, self._last_logits)
                ).item()
            else:
                pred = np.argmax(self._last_logits, axis=-1).item()
                reward = (
                    self.correct_reward
                    if pred == self.current_target
                    else self.incorrect_reward
                )

        return reward

    def step(self, action):
        obs, reward, done, info = super().step(action)

        info["target"] = self.current_target
        if done and action == self.num_features:
            info["classifier_was_correct"] = (
                np.argmax(self._last_logits) == self.current_target
            )
            info["classifier_logits"] = self._last_logits

        return obs, reward, done, info


class InstanceRecognitionEnv(AcquisitionEnv):
    """An acquisition environment where the goal is instance recognition.

    Note that this class is just a thin wrapper around `AcquisitionEnv` that specifies
    the number of actions (there is just one terminal action). Generally, this
    environment should be wrapped with a wrapper that will provide meaningful rewards.
    """

    def __init__(self, dataset_manager: EnvironmentDatasetManager, **kwargs):
        super().__init__(dataset_manager, **kwargs)

        self.action_space = gym.spaces.Discrete(self.num_features + 1)
