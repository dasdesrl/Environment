from dataclasses import dataclass, field

import gymnasium as gym

from .generation_system import GenerationSystem
from .storage_system import StorageSystem


@dataclass
class EnergySchedulingEnv(gym.Env):
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
        present = [battery.present_charge for battery in self.storage_system.batteries]
        capacities = [
            battery.CAPACITY_CHARGE for battery in self.storage_system.batteries
        ]
        info_dict = {"present_charge": present, "capacities": capacities}
        return info_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # uniform sample from observation_space
        sample = self.observation_space.sample()
        self.generation_system.current_state = sample["generation"]
        for battery, charge in zip(self.storage_system.batteries, sample["storage"]):
            battery.present_charge = charge

        initial_observation = self._get_obs()
        info = self._get_info()

        return initial_observation, info

    def step(self, action: gym.core.ActType) -> tuple[gym.core.ObsType,]:
        satisfied = self.constraint(action)
        assert satisfied, "Action did not conform to the constraints"

        reward, terminated = self.storage_system.step(action)
        self.generation_system.step()

        observation = self._get_obs()
        reward = reward
        terminated = terminated
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def constraint(self, action: gym.core.ActType):
        satisfied_storage = self.storage_system.constraint(action)
        satisfied_generation = self.generation_system.constraint(action)
        satisfied = satisfied_storage and satisfied_generation
        return satisfied


if __name__ == "__main__":
    env = EnergySchedulingEnv()

    observation_space = env.observation_space
    sample = env.observation_space.sample()
    assert observation_space.contains(sample)

    current_observation = env._get_obs()
    assert observation_space.contains(current_observation)

    from pprint import pprint

    pprint(env)
