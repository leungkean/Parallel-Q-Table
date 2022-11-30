from collections import defaultdict
from typing import Dict, Any, Tuple, Callable, List

import gym
import tensorflow as tf
from keras.metrics import Mean
from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.sgd import minibatches, standardized
from ray.util.ml_utils.dict import deep_update

from afa.agents.base import Agent
from afa.agents.utils import compute_gae_for_batch, explained_variance
from afa.policies.base import Policy
from afa.policies.model import ModelPolicy
from afa.rollouts import WorkerSet, aggregate_stat_dicts
from afa.typing import ConfigDict, Array, Observation

DEFAULT_CONFIG = {
    "model_config": {"value_network": "copy"},
    "num_workers": 8,
    "total_rollouts_length": 2048,
    "num_sgd_epochs": 16,
    "minibatch_size": 128,
    "learning_rate": 5e-5,
    # Coefficient of the entropy regularizer.
    "entropy_coef": 0.0,
    # Coefficient of the value function loss. IMPORTANT: this must be tuned this if
    # you set value_network="shared" inside your model's config.
    "vf_coef": 1.0,
    # PPO clip parameter.
    "clip_range": 0.2,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip": 10.0,
    "max_grad_norm": None,
    "gamma": 0.99,
    "lambda": 0.95,
    "evaluation_config": {},
}


class PPOAgent(Agent):
    """The Proximal Policy Optimization algorithm.

    This implementation uses the clipped surrogate objective.
    """

    def __init__(self, config: ConfigDict, env_fn: Callable[[ConfigDict], gym.Env]):
        super().__init__(config, env_fn)

        self._policy = ModelPolicy(
            self.observation_space, self.action_space, config["model_config"]
        )

        # The below attributes get initialized when train_setup is called.
        self._workers: WorkerSet = None
        self._evaluation_workers: WorkerSet = None
        self._optimizer: tf.keras.Optimizer = None

    @property
    def default_config(self) -> ConfigDict:
        return DEFAULT_CONFIG

    @property
    def policy(self) -> Policy:
        return self._policy

    def train_setup(self):
        self._workers = WorkerSet(
            num_workers=self.config["num_workers"],
            env_fn=self.env_fn,
            policy_cls=ModelPolicy,
            env_config=self.config["env_config"],
            policy_config=dict(model_config=self.config["model_config"]),
            batch_mode="truncate",
            rollout_length=(
                self.config["total_rollouts_length"] // self.config["num_workers"]
            ),
        )
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"]
        )

    @tf.function
    def _update(
        self,
        obs: Observation,
        actions: Array,
        advantages: Array,
        value_targets: Array,
        old_log_probs: Array,
    ):
        with tf.GradientTape() as tape:
            # Compute the policy and value predictions for the given observations
            pi, value_preds = self._policy.model(obs)
            # Retrieve policy entropy and the log probabilities of the actions
            log_probs = pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss
            ratio = tf.exp(log_probs - old_log_probs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(
                ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"]
            )
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
            # Define the value loss
            value_preds_clipped = tf.clip_by_value(
                value_preds, -self.config["vf_clip"], self.config["vf_clip"]
            )
            vf_loss_unclipped = (value_targets - value_preds) ** 2
            vf_loss_clipped = (value_targets - value_preds_clipped) ** 2
            value_loss = 0.5 * tf.reduce_mean(
                tf.maximum(vf_loss_clipped, vf_loss_unclipped)
            )
            # The final loss to be minimized is a combination of the policy and value
            # losses, in addition to an entropy bonus which can be used to encourage
            # exploration
            loss = (
                policy_loss
                - entropy * self.config["entropy_coef"]
                + value_loss * self.config["vf_coef"]
            )

        clip_frac = tf.reduce_mean(
            tf.cast(
                tf.greater(tf.abs(ratio - 1.0), self.config["clip_range"]), tf.float32
            )
        )

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self._policy.model.trainable_weights)
        # Perform gradient clipping
        if self.config["max_grad_norm"] is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.config["max_grad_norm"])
        # Apply the gradient update
        self._optimizer.apply_gradients(
            zip(grads, self._policy.model.trainable_weights)
        )

        # This is a measure of how well the value function explains the variance in
        # the rewards
        value_explained_variance = explained_variance(value_targets, value_preds)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "vf_explained_variance": value_explained_variance,
            "clip_frac": clip_frac,
        }

    def train_step(self, step: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._workers.set_weights(self.policy.get_weights())

        batches, stats = self._workers.sample()

        postproccessed_batches = [
            compute_gae_for_batch(
                self._policy, b, self.config["gamma"], self.config["lambda"]
            )
            for b in batches
        ]

        trajectory_data = SampleBatch.concat_samples(postproccessed_batches)
        trajectory_data[Postprocessing.ADVANTAGES] = standardized(
            trajectory_data[Postprocessing.ADVANTAGES]
        )

        metric_means = defaultdict(Mean)

        for _ in range(self.config["num_sgd_epochs"]):
            for batch in minibatches(trajectory_data, self.config["minibatch_size"]):
                metrics = self._update(
                    batch[SampleBatch.OBS],
                    batch[SampleBatch.ACTIONS],
                    batch[Postprocessing.ADVANTAGES],
                    batch[Postprocessing.VALUE_TARGETS],
                    batch[SampleBatch.ACTION_LOGP],
                )

                for k, v in metrics.items():
                    metric_means[k].update_state(v)

        metrics = {k: v.result() for k, v in metric_means.items()}

        return metrics, stats

    def evaluate(self, num_episodes: int) -> Tuple[Dict[str, Any], List[SampleBatch]]:
        if self._evaluation_workers is None:
            eval_config = deep_update(
                self.config, self.config["evaluation_config"], new_keys_allowed=True
            )

            self._evaluation_workers = WorkerSet(
                num_workers=eval_config["num_workers"],
                env_fn=self.env_fn,
                policy_cls=ModelPolicy,
                env_config=eval_config["env_config"],
                policy_config=dict(model_config=eval_config["model_config"]),
                batch_mode="complete",
            )

        stat_dicts, episodes = [], []

        self._evaluation_workers.set_weights(self.policy.get_weights())

        for _ in range(num_episodes // self._evaluation_workers.num_workers):
            eps, stats = self._evaluation_workers.sample(
                deterministic=self.config["evaluation_config"].get(
                    "deterministic", True
                )
            )
            stat_dicts.append(stats)
            episodes.extend(eps)

        return aggregate_stat_dicts(stat_dicts), episodes
