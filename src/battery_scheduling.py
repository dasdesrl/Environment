import gymnasium as gym
import numpy as np

from storage_system import StorageSystem
from generation_system import GenerationSystem


class BatterySchedulingEnv(gym.Env):
    def __init__(self):
        self.storage_system: StorageSystem = StorageSystem.default()
        self.generation_system: GenerationSystem = GenerationSystem.default()

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


if __name__ == "__main__":
    env = BatteryScheduling()
