import typing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from src.battery import Battery
from src.degradation import DegradationModel
from src.environment_typing import envt


@dataclass
class StorageSystem:
    batteries: typing.List[Battery]

    def observation_space(self) -> gym.spaces.Space[gym.core.ObsType]:
        obs_space = gym.spaces.Box(
            low=np.zeros((len(self.batteries),), dtype=envt.uint),
            high=np.array([battery.CAPACITY_CHARGE for battery in self.batteries]),
            dtype=envt.uint,
        )
        return obs_space

    def _get_obs(self) -> gym.spaces.Space[gym.core.ObsType]:
        battery_states = np.array(
            [battery.present_charge for battery in self.batteries]
        )
        return battery_states

    def _get_info(self):
        raise NotImplementedError

    def action_space(self) -> gym.spaces.Space[gym.core.ActType]:
        range_charges = np.array(
            [battery.able_charge() for battery in self.batteries],
        )
        able_discharges, able_charges = range_charges[:, 0], range_charges[:, 1]

        act_space = gym.spaces.Box(
            low=-able_discharges,
            high=able_charges,
            dtype=envt.int,
        )
        return act_space

    @classmethod
    def default(cls) -> typing.Self:
        num_batteries = 3
        batteries = [
            Battery(
                present_charge=50.0,
                degradation_model=DegradationModel(
                    alpha=1 / (i + 1), beta=1 / (2 * i + 1)
                ),
                CAPACITY_CHARGE=100.0 * (1 + i * 0.2),
                RATE_CHARGE=20.0 * (1 + i * 0.1),
                RATE_DISCHARGE=20.0 * (1 + i * 0.1),
            )
            for i in range(num_batteries)
        ]

        storage_system = cls(
            batteries=batteries,
        )
        return storage_system


if __name__ == "__main__":
    my_storage_system = StorageSystem.default()

    observation_space = my_storage_system.observation_space()
    sample = observation_space.sample()
    assert observation_space.contains(sample)

    from pprint import pprint

    pprint(my_storage_system)
