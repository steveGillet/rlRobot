import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, title="DQN Training", filename="dqn_rewards.png", window=50, color="tab:blue"):
    rewards = np.array(rewards)
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, color="lightgray", alpha=0.6, label="Raw reward")

    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        plt.plot(episodes[window-1:], smoothed, color=color, linewidth=2.5,
                 label=f"Rolling average ({window} episodes)")

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Episode reward", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved → {filename}")