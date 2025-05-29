import numpy as np

def simulate_demand(price, base_demand=100, price_sensitivity=2.0):
    """Returns units sold based on price."""
    demand = max(0, base_demand - price_sensitivity * (price - 10))
    return np.random.poisson(demand)

x = simulate_demand(10, 100, 2.0)
print(f"Simulated demand: {x}")
