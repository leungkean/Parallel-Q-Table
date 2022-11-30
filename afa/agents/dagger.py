from typing import Dict, Any, Callable, Tuple, Type, Optional, List

import gym
import numpy as np
import optax
import tensorflow as tf
from keras.metrics import Mean
from ray.rllib import SampleBatch
from ray.rllib.execution import ReplayBuffer
from ray.util.ml_utils.dict import deep_update
from tensorflow_probability import distributions as tfd
from tensorflow_probability.substrates.numpy.distributions import (
    Categorical,
    MixtureSameFamily,
)

from afa.agents.base import Agent
from afa.policies.base import Policy
from afa.policies.model import ModelPolicy
from afa.policies.utils import get_policy_cls
from afa.rollouts import WorkerSet, aggregate_stat_dicts
from afa.typing import Observation, NumpyDistribution, Weights, ConfigDict, Array
from afa.utils import get_schedule

DEFAULT_CONFIG = {
    "model_config": {},
    "oracle_type": "random_search",
    "oracle_config": {
        "max_observed": 0.5,
        "num_samples": 500,
        "fitness_fn_config": {
            "type": "classifier",
            "kwargs": {
                "terminal_reward_fn_config": {"type": "xent"},
                # The below keys need to be provided in the config when the agent
                # is created.
                # "classifier_path": ...,
                # "acquisition_cost": ...,
            },
        },
    },
    "beta_config": {
        "type": "exponential",
        "kwargs": {
            "init_value": 1.0,
            "decay_rate": 0.99,
            "transition_steps": 1,
        },
    },
    "buffer_capacity": 500000,
    "buffer_init": 5000,
    "num_workers": 1,
    "total_rollouts_length": 1024,
    "num_sgd_iters": 32,
    "minibatch_size": 32,
    "learning_rate": 0.001,
    "evaluation_config": {},
}


class DaggerAgent(Agent):
    """The DAgger imitation learning algorithm.

    In this version of DAgger, we store experience in a replay buffer and
    update the model on a set number of samples from the buffer per iteration.
    """

    def __init__(self, config: ConfigDict, env_fn: Callable[[], gym.Env]):
        super().__init__(config, env_fn)

        self._policy = ModelPolicy(
            self.observation_space, self.action_space, config["model_config"]
        )

        # The below attributes get initialized when train_setup is called.
        self._buffer: ReplayBuffer = None
        self._workers: WorkerSet = None
        self._evaluation_workers: WorkerSet = None
        self._beta_schedule: optax.Schedule = None
        self._optimizer: tf.keras.Optimizer = None

    @property
    def default_config(self) -> ConfigDict:
        return DEFAULT_CONFIG

    @property
    def policy(self) -> Policy:
        return self._policy

    def train_setup(self):
        self._buffer = ReplayBuffer(self.config["buffer_capacity"])
        self._workers = WorkerSet(
            num_workers=self.config["num_workers"],
            env_fn=self.env_fn,
            policy_cls=MixturePolicy,
            env_config=self.config["env_config"],
            policy_config=dict(
                oracle_policy_cls=get_policy_cls(self.config["oracle_type"]),
                oracle_policy_config=self.config["oracle_config"],
                learner_policy_cls=ModelPolicy,
                learner_policy_config=dict(model_config=self.config["model_config"]),
            ),
            batch_mode="truncate",
            rollout_length=(
                self.config["total_rollouts_length"] // self.config["num_workers"]
            ),
        )
        self._beta_schedule = get_schedule(self.config["beta_config"])
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"]
        )

    def evaluate(self, num_episodes: int) -> Tuple[Dict[str, Any], List[SampleBatch]]:
        if self._evaluation_workers is None:
            eval_config = deep_update(
                self.config, self.config["evaluation_config"], new_keys_allowed=True
            )

            # During evaluation, we only use the learned `ModelPolicy`, rather than
            # the mixture with the oracle.
            self._evaluation_workers = WorkerSet(
                num_workers=eval_config["num_workers"],
                env_fn=self.env_fn,
                policy_cls=ModelPolicy,
                env_config=eval_config.get("env_config", {}),
                policy_config=dict(model_config=self.config["model_config"]),
                batch_mode="complete",
            )

        stat_dicts, episodes = [], []

        self._evaluation_workers.set_weights(self.policy.get_weights())

        for _ in range(num_episodes):
            eps, stats = self._evaluation_workers.sample(
                deterministic=self.config["evaluation_config"].get(
                    "deterministic", True
                )
            )
            stat_dicts.append(stats)
            episodes.extend(eps)

        return aggregate_stat_dicts(stat_dicts), episodes

    def validate_config(self, config: ConfigDict) -> ConfigDict:
        config = super(DaggerAgent, self).validate_config(config)

        assert (
            config["total_rollouts_length"] % config["num_workers"] == 0
        ), "total_rollouts_length should be divisible by num_workers"

        return config

    @tf.function
    def _update_model(self, obs: Observation, oracle_probs: Array):
        oracle_pi = tfd.Categorical(probs=oracle_probs)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self._policy.model.trainable_weights)

            model_pi, _ = self._policy.model(obs, training=True)
            loss = tf.reduce_mean(oracle_pi.cross_entropy(model_pi))

        grads = tape.gradient(loss, self._policy.model.trainable_weights)
        self._optimizer.apply_gradients(
            zip(grads, self._policy.model.trainable_weights)
        )

        return loss

    def train_step(self, step: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._workers.set_weights(self.policy.get_weights())

        beta = self._beta_schedule(step)
        stat_dicts = []

        while True:
            batches, stats = self._workers.sample(beta=beta)
            stat_dicts.append(stats)

            for b in batches:
                for t in b.timeslices(1):
                    self._buffer.add(t, weight=1.0)

            # Don't start learning until the buffer has been filled up a given amount.
            if len(self._buffer) >= self.config["buffer_init"]:
                break

        loss_mean = Mean()

        for _ in range(self.config["num_sgd_iters"]):
            batch = self._buffer.sample(self.config["minibatch_size"])
            # obs = jax.tree_map(
            #     lambda *args: np.stack(args, axis=0), *batch[SampleBatch.OBS]
            # )
            loss = self._update_model(batch[SampleBatch.OBS], batch["oracle_probs"])
            loss_mean.update_state(loss)

        aggregate_stats = aggregate_stat_dicts(stat_dicts)

        metrics = {
            "loss": loss_mean.result().numpy(),
            "beta": beta,
        }

        return metrics, aggregate_stats


class MixturePolicy(Policy):
    """A mixture between two policies.

    This is used for experience collection when training a policy with DAgger. We mix
    between the oracle policy and the learned policy.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        oracle_policy_cls: Type[Policy],
        learner_policy_cls: Type[Policy],
        oracle_policy_config: Optional[Dict[str, Any]] = None,
        learner_policy_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(observation_space, action_space)

        self._oracle = oracle_policy_cls(
            observation_space, action_space, **(oracle_policy_config or {})
        )
        self._learner = learner_policy_cls(
            observation_space, action_space, **(learner_policy_config or {})
        )

    def compute_policy(
        self, obs: Observation, **kwargs
    ) -> Tuple[NumpyDistribution, Dict[str, Any]]:
        assert "beta" in kwargs, (
            "MixturePolicy requires that `beta` be passed to `compute_policy` "
            "as a keyword argument."
        )

        beta = kwargs["beta"]
        assert 0.0 <= beta <= 1.0, "`beta` must be in the range [0, 1]."

        oracle_pi, oracle_info = self._oracle.compute_policy(obs, **kwargs)
        learner_pi, learner_info = self._learner.compute_policy(obs, **kwargs)

        with np.errstate(divide="ignore"):
            oracle_logits = (
                oracle_pi.logits
                if oracle_pi.logits is not None
                else np.log(oracle_pi.probs)
            )
            learner_logits = (
                learner_pi.logits
                if learner_pi.logits is not None
                else np.log(learner_pi.probs)
            )

        stacked_logits = np.stack([oracle_logits, learner_logits], axis=0)
        components_distribution = Categorical(logits=stacked_logits)

        with np.errstate(divide="ignore"):
            mixture_logits = np.log([beta, 1 - beta])
            mixture_logits = np.where(mixture_logits == -np.inf, -1e12, mixture_logits)

        mixture_pi = MixtureSameFamily(
            mixture_distribution=Categorical(logits=mixture_logits),
            components_distribution=components_distribution,
        )

        oracle_probs = (
            oracle_pi.probs if oracle_pi.probs is not None else np.exp(oracle_logits)
        )

        extra_info = {"oracle_probs": oracle_probs}

        return mixture_pi, extra_info

    def get_weights(self) -> Weights:
        return self._learner.get_weights()

    def set_weights(self, weights: Weights):
        self._learner.set_weights(weights)
