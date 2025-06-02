import numpy as np
from stable_baselines3 import DQN
from gym_environment import DynamicPricingEnv
from collections import defaultdict
from dataset_feature_extraction import load_preprocess_data

# Run  dataset_feature_extraction function
df = load_preprocess_data()

# Initialize environment
env = DynamicPricingEnv(df)

# Create and train model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate trained model

action_rewards = defaultdict(list)

for episode in range(1):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)  # convert to int
        obs, reward, done, info = env.step(action)
        action_rewards[action].append(reward)


# Average the rewards per action
average_rewards = {action: np.mean(rewards) for action, rewards in action_rewards.items()}

# Print
for action, avg in sorted(average_rewards.items()):
    print(f"Action: {action} (Price: {10 + action}), Average Reward: {avg:.2f}")

best_action = max(average_rewards, key=average_rewards.get)
print(f"\n💰 Most profitable price based on average reward: ${10 + best_action}")
