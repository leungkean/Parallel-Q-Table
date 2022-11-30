import gym


class MonitorEpisodeWrapper(gym.Wrapper):
    """A wrapper that monitors episode data from an environment.

    When an episode has finished, an 'episode' entry is put into the info dict that
    contains the episode's length ('length') and total reward ('reward').

    Args:
        env: The environment being wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

        self._reward = 0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._reward = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._reward += reward
        self._ep_length += 1

        if done:
            ep_info = {"reward": round(self._reward, 6), "length": self._ep_length}
            info.setdefault("episode", {}).update(ep_info)

        return observation, reward, done, info
