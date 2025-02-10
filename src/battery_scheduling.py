import gymnasium as gym

from .storage_system import StorageSystem
from .generation_system import GenerationSystem

from dataclasses import dataclass, field


@dataclass
class BatterySchedulingEnv(gym.Env):
    storage_system: StorageSystem = field(default_factory=StorageSystem.default)
    generation_system: GenerationSystem = field(
        default_factory=GenerationSystem.default,
    )

    def __post_init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "storage": self.storage_system.observation_space(),
                "generation": self.generation_system.observation_space(),
            }
        )
        self.action_space = self.storage_system.action_space()

    def _get_obs(self):
        obs = {
            "storage": self.storage_system._get_obs(),
            "generation": self.generation_system._get_obs(),
        }
        return obs

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.storage_system.reset(seed, options)
        self.generation_system.reset(seed, options)

        initial_observation = self._get_obs()
        info = self._get_info()

        return initial_observation, info

    def step(self, action: gym.core.ActType) -> tuple[gym.core.ObsType,]:
        if self.constraints(action):
            pass
        else:
            reward = -1000

        return


if __name__ == "__main__":
    env = BatterySchedulingEnv()

    observation_space = env.observation_space
    sample = env.observation_space.sample()
    assert observation_space.contains(sample)

    current_observation = env._get_obs()
    assert observation_space.contains(current_observation)

    from pprint import pprint

    pprint(env)
