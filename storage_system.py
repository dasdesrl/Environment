import typing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .battery import Battery
from .degradation import DegradationModel
from .environment_typing import envt


@dataclass
class StorageSystem:
    batteries: typing.List[Battery]

    def observation_space(self) -> gym.spaces.Space[gym.core.ObsType]:
        # MultiDiscrete with `start`, Box could break in SB3
        obs_space = gym.spaces.Box(
            low=np.zeros((len(self.batteries),), dtype=envt.uint),
            high=np.array(
                [battery.CAPACITY_CHARGE for battery in self.batteries], dtype=envt.uint
            ),
            dtype=envt.uint,
        )
        return obs_space

    def _get_obs(self) -> gym.spaces.Space[gym.core.ObsType]:
        battery_states = np.array(
            [battery.present_charge for battery in self.batteries],
            dtype=envt.uint,
        )
        return battery_states

    def _get_info(self):
        action_space = self.action_space()
        low_action, high_action = action_space.low, action_space.high
        info_dict = {"low_action": low_action, "high_action": high_action}
        return info_dict

    def action_space(self) -> gym.spaces.Space[gym.core.ActType]:
        extremes_range = np.array(
            [
                [battery.CAPACITY_CHARGE, battery.CAPACITY_CHARGE]
                for battery in self.batteries
            ],
            dtype=envt.int,
        )
        extreme_discharges, extreme_charges = extremes_range[:, 0], extremes_range[:, 1]
        extreme_act_space = gym.spaces.Box(
            low=-extreme_discharges,
            high=+extreme_charges,
            dtype=envt.int,
        )
        return extreme_act_space

    def legality_penalty(self, action_vector: gym.core.ActType):
        constrained_act_space = self.action_mask()
        low_action, high_action = constrained_act_space.low, constrained_act_space.high

        num_batteries = len(self.batteries)
        clipped_action_vector = [
            np.clip(action_vector[i], low_action[i], high_action[i])
            for i in range(num_batteries)
        ]
        legal_separation_vector = action_vector - clipped_action_vector

        manhatten_norm = lambda sep: np.linalg.norm(sep, ord=1)

        how_far_from_legal_action = manhatten_norm(legal_separation_vector)
        return how_far_from_legal_action

    def constraint(self, action: gym.core.ActType):
        batteries_satisfied = [
            b_i.constraint(a_i) for b_i, a_i in zip(self.batteries, action)
        ]
        satisfied = np.all(batteries_satisfied)
        return satisfied

    def action_mask(self) -> gym.spaces.Space[gym.core.ActType]:
        range_charges = np.array(
            [battery.able_charge() for battery in self.batteries],
            dtype=envt.int,
        )
        able_discharges, able_charges = range_charges[:, 0], range_charges[:, 1]

        # MultiDiscrete with `start`, Box could break in SB3
        act_space = gym.spaces.Box(
            low=-able_discharges,
            high=able_charges,
            dtype=envt.int,
        )

        return act_space

    def step(self, action: gym.core.ActType):
        rewards = []
        for b_i, a_i in zip(self.batteries, action):
            initial_charge = b_i.present_charge
            b_i.charge_discharge(a_i)
            dod = abs(b_i.present_charge - initial_charge)
            dod_cycle_depth = dod / b_i.CAPACITY_CHARGE
            degradation = b_i.degradation_model.percent_degradation(dod_cycle_depth)
            rewards.append(-degradation)

        reward = np.sum(rewards)
        terminated = np.all([battery.present_charge == 0 for battery in self.batteries])

        return reward, terminated

    @classmethod
    def default(cls) -> typing.Self:
        num_batteries = 3
        batteries = [
            Battery(
                present_charge=5,
                degradation_model=DegradationModel(
                    alpha=1 / (i + 1), beta=1 / (2 * i + 1)
                ),
                CAPACITY_CHARGE=5 * (1 + i),
                RATE_CHARGE=2 * (1 + i),
                RATE_DISCHARGE=2 * (1 + i),
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
