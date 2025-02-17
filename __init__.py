from gymnasium.envs.registration import register

register(
    id="Environment/BatterySchedulingEnv-v0",
    entry_point="Environment.src:BatterySchedulingEnv"
)
