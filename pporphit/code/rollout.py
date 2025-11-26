from stable_baselines3 import PPO
import numpy as np

# Load the trained model
ppo = PPO.load("shelfArm")

# Dummy observation from your env
obs = np.array([0.0])

# Get the best (deterministic mean) action
best_action, _ = ppo.predict(obs, deterministic=True)

print("Best action (flat vector):", best_action)

# Decode it to meaningful morphology (using your env's step logic)
min_links = 2
max_links = 7
min_length = 0.05
max_length = 1.2

num_links = int(np.round(best_action[0] * (max_links - min_links) + min_links))
lengths = (best_action[1:(max_links + 1)] * (max_length - min_length) + min_length)[:num_links]
joint_types = np.round(best_action[(1 + max_links):] * 3)[:num_links].astype(int)

print("Decoded optimal morphology:")
print("Num Links:", num_links)
print("Lengths:", lengths)
print("Joint Types:", joint_types)