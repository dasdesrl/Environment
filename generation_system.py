import typing
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .environment_typing import envt


@dataclass
class GenerationSystem:
    current_state: envt.uint
    R_VALUES: np.typing.NDArray
    TRANSITION_MATRIX: np.typing.NDArray

    def observation_space(self):
        obs_space = gym.spaces.Discrete(
            n=envt.uint(len(self.R_VALUES)),
        )
        return obs_space

    def _get_obs(self):
        return self.current_state

    def _get_net_generation(self):
        net_generation = self.R_VALUES[self.current_state]
        return net_generation

    def _get_info(self):
        info_dict = {"net_generation": self._get_net_generation()}
        return info_dict

    def step(self):
        next_state = np.random.choice(
            len(self.R_VALUES), p=self.TRANSITION_MATRIX[self.current_state]
        )
        self.current_state = next_state
        return

    def legality_penalty(self, action: gym.core.ActType):
        current_generation = self._get_net_generation()
        total_of_action = sum(action)
        net_difference = current_generation - total_of_action

        how_far_from_sum_of_actions_equals_generation = abs(net_difference)
        return how_far_from_sum_of_actions_equals_generation

    def constraint(self, action: gym.core.ActType):
        current_net_generation = self._get_net_generation()
        satisfied = np.sum(action) == current_net_generation
        if not satisfied:
            # print(f"Action did not satisfy generation constraint sum({action}) clippedby generation {current_net_generation}")
            pass
        return satisfied

    def action_mask(self, possible_actions):
        current_generation = self._get_net_generation()
        mask = np.sum(possible_actions, axis=1) == current_generation

        return mask

    @classmethod
    def default(cls) -> typing.Self:
        # paper: "The state space $\mathcal{S}_𝑟$ contains all the possible
        # values which the supply minus demand process can take. "
        # R_VALUES = np.arange(start=-50, stop=50, step=1, dtype=envt.int)
        R_VALUES = np.array([-4, -1, 1, 5], dtype=envt.int)
        current_state = 0
        TRANSITION_MATRIX = cls._create_transition_matrix(len(R_VALUES))

        generation_system = cls(
            current_state=current_state,
            R_VALUES=R_VALUES,
            TRANSITION_MATRIX=TRANSITION_MATRIX,
        )
        return generation_system

    @staticmethod
    def _create_transition_matrix(n_states) -> np.typing.NDArray:
        P = np.zeros((n_states, n_states), dtype=envt.float)

        # Helper to compute distance between states
        def state_distance(i, j):
            i_x, i_y = i // 10, i % 10
            j_x, j_y = j // 10, j % 10
            return np.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2)

        # Fill transition probabilities
        for i in range(n_states):
            distances = np.array([state_distance(i, j) for j in range(n_states)])
            # probs = np.exp(-0.5 * distances)  # Exponential decay with distance
            probs = np.exp(-0.01 * distances)

            probs /= np.sum(probs)  # Normalize
            P[i] = probs
        # assert np.all(np.sum(P, axis=1)==1), "GenerationSystemError: Transition matrix rows do not sum up to 1"

        return P


if __name__ == "__main__":
    my_generation_system = GenerationSystem.default()

    observation_space = my_generation_system.observation_space()
    sample = observation_space.sample()
    assert observation_space.contains(sample)

    from pprint import pprint

    pprint(my_generation_system)
