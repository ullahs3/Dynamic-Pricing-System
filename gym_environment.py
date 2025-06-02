import gym
from gym import spaces
import numpy as np

def simulate_demand(price, base_demand=100, price_sensitivity=2.0):
    """Returns units sold based on price."""
    demand = max(0, base_demand - price_sensitivity * (price - 10))
    return np.random.poisson(demand)


class DynamicPricingEnv(gym.Env):
    def __init__(self, data):
        super(DynamicPricingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.max_inventory = 100
        self.inventory = self.max_inventory
        self.cost = 10

        self.action_space = spaces.Discrete(11)  # Actions = price levels from $10 to $20
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.inventory = self.max_inventory
        return self._get_obs()

    def _get_obs(self):
        obs = self.data.iloc[self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        price = 10 + action

        # Simulate demand from price and base demand
        demand = max(0, 100 - 2 * (price - 10))  # base_demand - sensitivity * (price - base)
        sold = min(np.random.poisson(demand), self.inventory)

        reward = (price - self.cost) * sold
        self.inventory -= sold

        self.current_step += 1
        done = self.current_step >= len(self.data) or self.inventory <= 0

        return self._get_obs(), reward, done, {}
