import gym
from gym import spaces
import numpy as np

def simulate_demand(price, base_demand=100, price_sensitivity=2.0):
    """Returns units sold based on price."""
    demand = max(0, base_demand - price_sensitivity * (price - 10))
    return np.random.poisson(demand)

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.action_space = spaces.Discrete(11)  # Prices from 10 to 20
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
        self.inventory = 100
        self.cost = 5

    def reset(self):
        self.inventory = 100
        return self._get_obs()

    def step(self, action):
        price = 10 + action
        demand = simulate_demand(price)
        sold = min(demand, self.inventory)
        reward = (price - self.cost) * sold
        self.inventory -= sold
        done = self.inventory <= 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([self.inventory, self.cost, np.random.rand()], dtype=np.float32)
