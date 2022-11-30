from typing import List, Dict

import numpy as np


def summarize_episode_infos(episode_infos: List[Dict]) -> Dict:
    data = {
        "episode_length": np.mean([d["length"] for d in episode_infos]).item(),
        "episode_reward": np.mean([d["reward"] for d in episode_infos]).item(),
    }

    if "sse" in episode_infos[0]:
        data["rmse"] = np.sqrt(np.mean([d["sse"] for d in episode_infos])).item()

    if "accuracy" in episode_infos[0]:
        data["accuracy"] = np.mean([d["accuracy"] for d in episode_infos])

    return data
