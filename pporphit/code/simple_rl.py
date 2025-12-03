import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from robotArmEnv import robotArmEnv
import time

# --- Configuration ---
NUM_EPISODES = 1000
LEARNING_RATE = 0.01
NUM_LINKS = 4
MIN_LENGTH = 0.05  # Environment constraint
MAX_LENGTH = 1.2


class PolicyNetwork(nn.Module):
    """
    Simple policy network that outputs mean and log_std for each link length.
    """

    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.sigmoid(self.mean(x))  # Sigmoid to keep in [0, 1]
        std = torch.exp(self.log_std)
        return mean, std


def get_env_action(lengths):
    """
    Converts link lengths to the 15-dim vector expected by robotArmEnv.
    """
    env_action = np.zeros(15, dtype=np.float32)

    # Set Num Links (normalized for 4 links)
    env_action[0] = 0.4

    # Set Lengths (already normalized to [0, 1])
    for i, length in enumerate(lengths):
        env_action[1 + i] = length

    # Set Joint Types (Z-Y-Y-Y pattern)
    fixed_joints = [2, 1, 1, 1]
    for i, j_type in enumerate(fixed_joints):
        env_action[8 + i] = j_type / 3.0

    return env_action


def train_reinforce():
    print("Initializing Environment...")
    env = robotArmEnv()

    # State dimension is 1 (dummy state [0.0])
    # Action dimension is 4 (one length per link, normalized [0, 1])
    state_dim = 1
    action_dim = NUM_LINKS

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    print(f"Starting REINFORCE for {NUM_EPISODES} episodes...")
    print(
        f"Optimizing {NUM_LINKS} link lengths between {MIN_LENGTH}m and {MAX_LENGTH}m"
    )
    print("-" * 50)

    for episode in range(NUM_EPISODES):
        # Get state (always [0.0] for this bandit problem)
        state = torch.FloatTensor([0.0])

        # Sample action from policy
        mean, std = policy(state)
        dist = torch.distributions.Normal(mean, std)
        action_normalized = dist.sample()  # Sample from Gaussian
        action_normalized = torch.clamp(action_normalized, 0.0, 1.0)  # Clip to [0, 1]
        log_prob = dist.log_prob(action_normalized).sum()  # Log probability

        # Convert to environment action
        env_action = get_env_action(action_normalized.detach().numpy())

        # Step environment
        obs, reward, done, truncated, info = env.step(env_action)

        # REINFORCE update: gradient = log_prob * reward
        # Since this is a one-step episode (bandit), no discounting needed
        loss = -log_prob * reward  # Negative because we want to maximize reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if (episode + 1) % 50 == 0:
            # Convert normalized action back to real lengths for display
            real_lengths = (
                action_normalized.detach().numpy() * (MAX_LENGTH - MIN_LENGTH)
                + MIN_LENGTH
            )
            print(
                f"Episode {episode+1}: Reward={reward:.2f}, Lengths={np.round(real_lengths, 3)}"
            )

    print("-" * 50)
    print("Training Complete.")

    # Evaluate final policy (deterministic = use mean)
    state = torch.FloatTensor([0.0])
    mean, _ = policy(state)
    final_lengths_normalized = mean.detach().numpy()
    final_lengths = final_lengths_normalized * (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH

    print(f"Final Policy (Mean Lengths): {np.round(final_lengths, 3)}")

    # Save model
    torch.save(policy.state_dict(), "reinforce_policy.pth")
    print("Policy saved to reinforce_policy.pth")


if __name__ == "__main__":
    train_reinforce()
