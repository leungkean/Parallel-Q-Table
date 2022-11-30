import itertools
import os
import random
import time
from collections import defaultdict
from typing import (
    Callable,
    Optional,
    Union,
    Type,
    Any,
    Mapping,
    Tuple,
    List,
    Dict,
    Iterable,
)

import gym
import numpy as np
import ray
from ray.rllib import SampleBatch
from tensorflow_probability.substrates.numpy.distributions import Categorical

from afa.policies.base import Policy
from afa.typing import Weights, ConfigDict
from afa.wrappers import MonitorEpisodeWrapper


class SampleCollector(dict):
    """Utility for aggregating trajectory data and converting to `SampleBatch`."""

    def add(self, key: str, value: Any):
        self.setdefault(key, []).append(value)

    def add_from(self, entries: Mapping[str, Any]):
        for k, v in entries.items():
            self.add(k, v)

    def build(self, ensure_single_episode: bool = True) -> SampleBatch:
        batch_data = dict()
        length = None

        for k, v in self.items():
            if length is None:
                length = len(v)
            else:
                assert length == len(v)

            # If we have dict observations, convert the list of
            # dicts to a dict of lists (or really, arrays).
            if k == SampleBatch.OBS and isinstance(v[0], dict):
                d = defaultdict(list)

                for e in v:
                    for k_, v_ in e.items():
                        d[k_].append(v_)

                for k_, v_ in d.items():
                    d[k_] = np.asarray(v_)

                v = d

            batch_data[k] = v

        if ensure_single_episode:
            assert (
                len(set(batch_data[SampleBatch.EPS_ID])) == 1
            ), "Batch spans more than one episode."

        return SampleBatch(batch_data)


class RolloutWorker:
    """A worker responsible for rolling out a policy in an environment.

    Args:
        env_fn: A function that accepts a configuration dictionary and returns an
            instance of the environment to collect trajectories from.
        env_config: The config dict to be passed to `env_fn`.
        policy_cls: The type of policy to use.
        policy_config: A dictionary containing keyword arguments to be passed to the
            policy upon creation.
        batch_mode: Either "truncate" or "complete":
            - "truncate": Each call to `sample` will always collect exactly
                `rollout_length` steps, even if this requires crossing episode
                boundaries.
            - "complete": Each call to `sample` will collect exactly one episode.
                Note that in this case, each call to `sample` may produce a different
                amount of experience.
        rollout_length: The number of steps to collect when `batch_mode="truncate"`.
        seed: The random seed used to initialize the environment.
    """

    def __init__(
        self,
        env_fn: Callable[[ConfigDict], gym.Env],
        policy_cls: Type[Policy],
        env_config: Optional[ConfigDict] = None,
        policy_config: Optional[ConfigDict] = None,
        batch_mode: str = "complete",
        rollout_length: int = 200,
        seed: Optional[int] = None,
    ):
        self.env = MonitorEpisodeWrapper(env_fn(env_config or {}))
        self.batch_mode = batch_mode
        self.rollout_length = rollout_length
        if self.batch_mode not in ["truncate", "complete"]:
            raise ValueError("batch_mode must be either 'truncate' or 'complete'")

        seed = seed or random.randrange(int(2e9))
        self.env.seed(seed)

        self.policy = policy_cls(
            self.env.observation_space, self.env.action_space, **(policy_config or {})
        )

        self.obs = self.env.reset()
        self.eps_id = random.randrange(int(2e9))

        print(f"Created RolloutWorker with PID {os.getpid()}")

    @classmethod
    def as_remote(
        cls,
        num_cpus: Optional[int] = None,
        num_gpus: Union[int, float] = 0,
        memory: Optional[int] = None,
        object_store_memory: Optional[int] = None,
        resources: Optional[dict] = None,
    ) -> Type:
        """Returns RolloutWorker class as a `@ray.remote using given options`.

        The returned class can then be used to instantiate ray actors.

        Args:
            num_cpus: The number of CPUs to allocate for the remote actor.
            num_gpus: The number of GPUs to allocate for the remote actor.
                This could be a fraction as well.
            memory: The heap memory request for the remote actor.
            object_store_memory: The object store memory for the remote actor.
            resources: The default custom resources to allocate for the remote
                actor.

        Returns:
            The `@ray.remote` decorated RolloutWorker class.
        """
        # Hide TensorFlow warnings for actors, e.g. so that we don't get CUDA warnings
        # when actors aren't using GPUs.
        runtime_env = {"env_vars": {"TF_CPP_MIN_LOG_LEVEL": "3"}}

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
            runtime_env=runtime_env,
        )(cls)

    def set_weights(self, weights: Weights):
        """Sets the weights of this worker's policy.

        Args:
            weights: The weights of the policy.

        Returns:
            None.
        """
        self.policy.set_weights(weights)

    def sample(
        self, deterministic: bool = False, **kwargs
    ) -> Tuple[List[SampleBatch], Dict[str, Any]]:
        """Samples experience from the environment by rolling out the policy.

        Args:
            deterministic: If false, actions will be sampled from the policy. Otherwise,
                the mode/mean of the policy will be used.
            **kwargs: Additional keyword arguments to be passed to the policy's
                `compute_policy` method.

        Returns:
            batches: A list of `SampleBatch`s containing the collected data, each of
                which contains a single episode.
            stats: Other stats about the sampling, e.g. average time per policy eval.
        """
        is_finished = False

        collector = SampleCollector()
        batches = []
        policy_eval_ms = []
        ep_infos = []

        # Make ground truth available to policy. This is required for some policies
        # such as "RandomSearchPolicy".
        if hasattr(self.env.unwrapped, "current_example"):
            kwargs["current_example"] = self.env.unwrapped.current_example
        if hasattr(self.env.unwrapped, "current_target"):
            kwargs["current_target"] = self.env.unwrapped.current_target

        for step in itertools.count(start=1):
            collector.add(SampleBatch.OBS, self.obs.copy())
            collector.add(SampleBatch.EPS_ID, self.eps_id)

            start = time.perf_counter()
            pi, extra_info = self.policy.compute_policy(self.obs, **kwargs)
            end = time.perf_counter()
            policy_eval_ms.append((end - start) * 1000)

            if deterministic:
                if isinstance(pi, Categorical):
                    action = pi.mode()
                else:
                    action = pi.mean()
            else:
                action = pi.sample()

            collector.add(SampleBatch.ACTIONS, action)
            collector.add(SampleBatch.ACTION_LOGP, pi.log_prob(action))
            collector.add_from(extra_info)

            self.obs, reward, done, info = self.env.step(action)
            next_obs = self.obs.copy()

            if done:
                self.eps_id = random.randrange(int(2e9))
                self.obs = self.env.reset()

                if hasattr(self.env.unwrapped, "current_example"):
                    kwargs["current_example"] = self.env.unwrapped.current_example
                if hasattr(self.env.unwrapped, "current_target"):
                    kwargs["current_target"] = self.env.unwrapped.current_target

                if self.batch_mode == "complete":
                    is_finished = True

            maybe_ep_info = info.pop("episode", None)
            if maybe_ep_info is not None:
                if "classifier_was_correct" in info:
                    maybe_ep_info["accuracy"] = info["classifier_was_correct"]
                ep_infos.append(maybe_ep_info)

            collector.add(SampleBatch.REWARDS, reward)
            collector.add(SampleBatch.DONES, done)
            collector.add(SampleBatch.NEXT_OBS, next_obs)
            collector.add(SampleBatch.INFOS, info)

            # See if we have transitioned to a new episode
            if self.eps_id != collector[SampleBatch.EPS_ID][-1]:
                batches.append(collector.build())
                collector.clear()

            if self.batch_mode == "truncate" and step == self.rollout_length:
                is_finished = True

            if is_finished:
                if collector:
                    batches.append(collector.build())
                break

        stats = {
            "mean_policy_eval_ms": sum(policy_eval_ms) / len(policy_eval_ms),
            "num_timesteps": sum(b.agent_steps() for b in batches),
            "num_episodes": len(batches),
            "episode_infos": ep_infos,
        }

        return batches, stats


class WorkerSet:
    """A set of multiple `RolloutWorker`s.

    Args:
        num_workers: The number of workers to create. If 0, then one worker will be
            created locally. Otherwise, the workers will be created remotely
            in their own processes.
        env_fn: A function that accepts a configuration dictionary and returns an
            instance of the environment to collect trajectories from.
        env_config: The config dict to be passed to `env_fn`.
        policy_cls: The type of policy to use.
        policy_config: A dictionary containing keyword arguments to be passed to the
            policy upon creation.
        batch_mode: Either "truncate" or "complete":
            - "truncate": Each call to `sample` will always collect exactly
                `rollout_length` steps per worker, even if this requires crossing
                episode boundaries.
            - "complete": Each call to `sample` will collect exactly one episode per
                worker. Note that in this case, each call to `sample` may produce a
                different amount of experience.
        rollout_length: The number of steps to collect from each worker when
            `batch_mode="truncate"`.
        seed: The random seed used to initialize the environments.
    """

    def __init__(
        self,
        num_workers: int,
        env_fn: Callable[[ConfigDict], gym.Env],
        policy_cls: Type[Policy],
        env_config: Optional[ConfigDict] = None,
        policy_config: Optional[ConfigDict] = None,
        batch_mode: str = "complete",
        rollout_length: int = 200,
        seed: Optional[int] = None,
    ):
        if num_workers == 0:
            self._local_worker = RolloutWorker(
                env_fn,
                policy_cls,
                env_config,
                policy_config,
                batch_mode,
                rollout_length,
                seed,
            )
            self._remote_workers = None
        else:
            seed = seed or random.randrange(int(2e9))

            self._local_worker = None
            self._remote_workers = [
                RolloutWorker.as_remote(num_gpus=0).remote(
                    env_fn,
                    policy_cls,
                    env_config,
                    policy_config,
                    batch_mode,
                    rollout_length,
                    seed + i,
                )
                for i in range(num_workers)
            ]

        self._num_workers = max(num_workers, 1)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def set_weights(self, weights: Weights):
        """Broadcasts weights to all of the workers' policies.

        Args:
            weights: The weights of to broadcast.

        Returns:
            None.
        """
        if self._local_worker is not None:
            self._local_worker.set_weights(weights)
        else:
            ray.get([w.set_weights.remote(weights) for w in self._remote_workers])

    def sample(
        self, deterministic: bool = False, **kwargs
    ) -> Tuple[List[SampleBatch], Dict[str, Any]]:
        """Samples experience from the environment by rolling out the policy.

        Args:
            deterministic: If false, actions will be sampled from the policy. Otherwise,
                the mode/mean of the policy will be used.
            **kwargs: Additional keyword arguments to be passed to the policy's
                `compute_policy` method.

        Returns:
            batches: A list of `SampleBatch`s containing the collected data, each of
                which contains a single episode.
            stats: Other stats about the sampling, e.g. average time per policy eval.
        """
        if self._local_worker is not None:
            return self._local_worker.sample(deterministic, **kwargs)

        batches, stats = zip(
            *ray.get(
                [w.sample.remote(deterministic, **kwargs) for w in self._remote_workers]
            )
        )

        batches = list(itertools.chain.from_iterable(batches))

        return batches, aggregate_stat_dicts(stats)


def aggregate_stat_dicts(stats: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Utility for aggregating the stats dicts from multiple workers.

    Args:
        stats: A list of stats dicts returned from `RolloutWorker.sample`.

    Returns:
        A single stats dict that is the aggregate of the provided ones.
    """
    return {
        "mean_policy_eval_ms": (
            sum(d["mean_policy_eval_ms"] for d in stats) / len(stats)
        ),
        "num_timesteps": sum(d["num_timesteps"] for d in stats),
        "num_episodes": sum(d["num_episodes"] for d in stats),
        "episode_infos": list(
            itertools.chain.from_iterable(d["episode_infos"] for d in stats)
        ),
    }
