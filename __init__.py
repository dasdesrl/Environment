from environment.energy_scheduling_env import EnergySchedulingEnv

from gymnasium.envs.registration import register

register(id="EnergySchedulingEnv-v0", entry_point="environment:EnergySchedulingEnv")
