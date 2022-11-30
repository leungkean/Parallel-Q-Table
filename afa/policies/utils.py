from typing import Type

from afa.policies.base import Policy
from afa.policies.random import RandomPolicy
from afa.policies.random_search import RandomSearchPolicy

_POLICIES = {
    "random": RandomPolicy,
    "random_search": RandomSearchPolicy,
}


def get_policy_cls(policy: str) -> Type[Policy]:
    """Looks up a policy class by name.

    Args:
        policy: The name of the policy class to retrieve.

    Returns:
        The class of the requested policy.
    """
    if policy not in _POLICIES:
        raise ValueError(f"{policy} is not a valid policy.")

    return _POLICIES[policy]
