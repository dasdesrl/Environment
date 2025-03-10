from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

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
        self.action_space = (
            self.storage_system.action_space()
        )  # what is the action space

    def _get_obs(self):
        obs = {
            "storage": self.storage_system._get_obs(),
            "generation": self.generation_system._get_obs(),
        }
        return obs

    def _get_info(self):
        storage_info_dict = self.storage_system._get_info()
        generation_info_dict = self.generation_system._get_info()
        info_dict = storage_info_dict | generation_info_dict
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
        if not satisfied:
            observation = self._get_obs()
            reward = -(
                self.storage_system.legality_penalty(action)
                + self.generation_system.legality_penalty(action)
            )
            terminated = False
            truncated = False
            info = self._get_info()

            return observation, reward, terminated, truncated, info

        reward, terminated = self.storage_system.step(action)

        present_charges = [
            battery.present_charge for battery in self.storage_system.batteries
        ]

        info_dict = self._get_info()
        net_generation = info_dict["actual_generation"]
        no_charge_left = sum(present_charges) < -net_generation
        no_capacity_left = (
            sum(self.storage_system.observation_space.high - present_charges)
            < net_generation
        )
        terminated = terminated or (no_charge_left or no_capacity_left)
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

    def action_masks(self):
        storage_constrained_action_space = self.storage_system.action_mask()

        low_action, high_action = (
            storage_constrained_action_space.low,
            storage_constrained_action_space.high,
        )
        storage_constrained_grid = np.meshgrid(
            *[np.arange(i, j + 1) for i, j in zip(low_action, high_action)]
        )
        storage_constrained_possible_actions = np.vstack(
            list(map(np.ravel, storage_constrained_grid))
        ).T

        generation_mask = self.generation_system.action_mask(
            storage_constrained_possible_actions
        )

        masked_actions = storage_constrained_possible_actions[generation_mask]
        return masked_actions


if __name__ == "__main__":
    env = EnergySchedulingEnv()

    observation_space = env.observation_space
    sample = env.observation_space.sample()
    assert observation_space.contains(sample)

    current_observation = env._get_obs()
    assert observation_space.contains(current_observation)

    from pprint import pprint

    pprint(env)
