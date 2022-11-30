import tensorflow as tf
from ray.rllib import SampleBatch
from ray.rllib.evaluation import compute_advantages
from ray.rllib.policy.view_requirement import ViewRequirement
from tensorflow import Tensor

from afa.policies.model import ModelPolicy
from afa.typing import Array


def compute_gae_for_batch(
    policy: ModelPolicy,
    batch: SampleBatch,
    gamma: float = 0.99,
    lambda_: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
):
    """Given a sample batch, compute its value targets and the advantages.

    Args:
        policy: The agent's model policy, which should have a value prediction output.
        batch: SampleBatch of a single trajectory (not necessarily a complete episode
            though).
        gamma: Discount factor.
        lambda_: Parameter for GAE.
        use_gae: Whether or not to use Generalized Advantage Estimation (or just plain
            advantages).
        use_critic: Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch that has been updated with advantages.
    """
    if batch[SampleBatch.DONES][-1]:
        # If this episode is actually complete, then the final reward is just zero,
        # i.e. there is no bootstrapping.
        last_r = 0.0
    else:
        # Otherwise, the final timestep is not terminal, so we bootstrap the final
        # reward.
        final_obs = batch.get_single_step_input_dict(
            {SampleBatch.OBS: ViewRequirement()}, index="last"
        )[SampleBatch.OBS][0]
        _, info = policy.compute_policy(final_obs)
        last_r = info[SampleBatch.VF_PREDS]

    return compute_advantages(batch, last_r, gamma, lambda_, use_gae, use_critic)


def explained_variance(targets: Array, preds: Array) -> Tensor:
    """Computes the explained variance between predictions and targets.

    Values closer to 1.0 mean that the targets and predictions are highly correlated.

    Args:
        targets: The target values.
        preds: The predicted values.

    Returns:
        The scalar percentage of variance in targets that is explained by preds.
    """
    y_var = tf.math.reduce_variance(targets, axis=0)
    diff_var = tf.math.reduce_variance(targets - preds, axis=0)
    return tf.maximum(-1.0, 1 - (diff_var / y_var))
