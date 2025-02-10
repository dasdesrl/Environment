from battery_scheduling import BatterySchedulingEnv

if __name__ == "__main__":
    env = BatterySchedulingEnv()
    SEED = 4
    env.reset(seed=SEED)
    from pprint import pprint

    pprint(env)
