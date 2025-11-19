import gymnasium as gym
import minigrid
from dqn_agent import DQNAgent
import numpy as np
from plots import plot_rewards

class FlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(147,), dtype=np.float32
        )

    def observation(self, obs):
        return obs["image"].astype(np.float32).reshape(-1) / 10.0

env = gym.make("MiniGrid-Dynamic-Obstacles-8x8-v0")
env = FlatObsWrapper(env)

print(env.observation_space)

agent = DQNAgent(state_size=147, action_size=3, seed=42,
                 lr=1e-3, batch_size=64, update_every=100, double_dqn=True)

eps = 1.0
eps_min = 0.01
eps_decay = 0.995

episode_rewards = []

for episode in range(1, 2001):         
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    eps = max(eps_min, eps_decay * eps)

    if episode % 50 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.3f} | Eps: {eps:.3f}")

plot_rewards(
    episode_rewards,
    title="2.4 Double DQN",
    filename="plot_2.4_ddqn.png",
    color="tab:red"
)