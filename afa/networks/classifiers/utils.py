from typing import Callable, Union, Tuple, List

import numpy as np
from tensorflow import saved_model

from afa.typing import Observation


def get_environment_classifier_fn(
    classifier_path: str, return_call_counter: bool = False
) -> Union[
    Callable[[Observation], np.ndarray],
    Tuple[Callable[[Observation], np.ndarray], List[int]],
]:
    classifier = saved_model.load(classifier_path)

    if return_call_counter:
        counter = [0]

        def classifier_fn(obs):
            counter[0] += 1
            obs["x"] = np.expand_dims(obs.pop("observed"), axis=0)
            obs["b"] = np.expand_dims(obs.pop("mask"), axis=0)
            return np.squeeze(classifier(obs), 0)

        return classifier_fn, counter

    def classifier_fn(obs):
        obs["x"] = np.expand_dims(obs.pop("observed"), axis=0)
        obs["b"] = np.expand_dims(obs.pop("mask"), axis=0)
        return np.squeeze(classifier(obs), 0)

    return classifier_fn
