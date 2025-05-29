from stable_baselines3 import DQN
from gym_environment import DynamicPricingEnv
from collections import Counter

# Initialize environment
env = DynamicPricingEnv()

# Create and train model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate trained model

action_counts = Counter()

for episode in range(1):  # Run 100 evaluation episodes
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_counts[action] += 1
        obs, reward, done, info = env.step(action)

# Print most common actions
for action, count in sorted(action_counts.items()):
    print(f"Action: {action} (Price: {10 + action}), Count: {count}")

most_common_action = action_counts.most_common(1)[0][0]
print(f"\n💰 Most profitable price the agent picked most often: ${10 + most_common_action}")
