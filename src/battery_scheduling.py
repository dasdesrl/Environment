import gymnasium as gym

from storage_system import StorageSystem
from generation_system import GenerationSystem

from dataclasses import dataclass, field

@dataclass
class BatterySchedulingEnv(gym.Env):
    storage_system: StorageSystem = field(default_factory=StorageSystem.default)
    generation_system: GenerationSystem = field(default_factory=GenerationSystem.default)
    
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


if __name__ == "__main__":
    env = BatteryScheduling()
