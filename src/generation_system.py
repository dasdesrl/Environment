from dataclasses import dataclass
import typing

import gymnasium as gym
import numpy as np


dtype = np.uint
dtype_int = np.int32


@dataclass
class GenerationSystem:
    current_state: dtype
    R_VALUES: np.typing.NDArray
    TRANSITION_MATRIX: np.typing.NDArray

    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=self.R_VALUES.min(),
            high=self.R_VALUES.max(),
            dtype=dtype_int,
        )
        return obs_space

    def _get_obs(self):
        state = self.R_VALUES[self.current_state]
        return np.array(
            [
                state,
            ],
            dtype=dtype_int,
        )

    @classmethod
    def default(cls) -> typing.Self:
        # paper: "The state space $\mathcal{S}_𝑟$ contains all the possible
        # values which the supply minus demand process can take. "
        R_VALUES = np.arange(start=-50, stop=50, step=1, dtype=dtype_int)
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
        P = np.zeros((n_states, n_states))

        # Helper to compute distance between states
        def state_distance(i, j):
            i_x, i_y = i // 10, i % 10
            j_x, j_y = j // 10, j % 10
            return np.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2)

        # Fill transition probabilities
        for i in range(n_states):
            distances = np.array([state_distance(i, j) for j in range(n_states)])
            probs = np.exp(-0.5 * distances)  # Exponential decay with distance
            probs = probs / sum(probs)  # Normalize
            P[i] = probs

        return P


if __name__ == "__main__":
    my_generation_system = GenerationSystem.default()

    observation_space = my_generation_system.observation_space()
    sample = observation_space.sample()
    assert observation_space.contains(sample)

    from pprint import pprint

    pprint(my_generation_system)
