from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import simTest  # Your env file
import matplotlib.pyplot as plt
import numpy as np
import time

# Config: quick_test=True for fast validation (~1-2 min)
quick_test = True  # Set to False for full

# Training env
n_envs = 8 if not quick_test else 1
env = make_vec_env(simTest.ManipulatorEnv, n_envs=n_envs)

# PPO
start_time = time.time()
timesteps = 1000 if quick_test else 100000
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, clip_range=0.2, seed=42)
model.learn(total_timesteps=timesteps, progress_bar=True)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Save early
model.save("ppo_morphology")
print("Model saved to ppo_morphology.zip in ~/Desktop/rlRobot/")

# Plot fix: Use name_to_value dict
try:
    rew_mean = model.logger.name_to_value.get("rollout/ep_rew_mean", [])
    plt.plot(rew_mean)
    plt.xlabel("Steps"); plt.ylabel("Mean Reward"); plt.savefig("ppo_learning_curve.png")
    plt.show()
    print("Plot saved to ppo_learning_curve.png")
except Exception as e:
    print(f"Plot error: {e} (no data yet?)")

# Eval with try-except
try:
    eval_env = simTest.ManipulatorEnv()
    n_episodes = 10 if quick_test else 100
    success_rates = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        success_rates.append(1 if terminated else 0)
    print(f"Avg Success Rate: {np.mean(success_rates):.2f}")
except Exception as e:
    print(f"Eval error: {e} (model still saved)")