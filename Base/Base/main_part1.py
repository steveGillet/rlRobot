# main.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from sarsa_agent import SarsaAgent
from q_learning_agent import QLearningAgent

def train(agent, env, episodes=200):
    """
    A simple training loop for a given agent and environment.
    """
    rewards_per_episode = []
    
    for episode in range(episodes):
        if (episode + 1) % 100 == 0:
            print(f"Training episode {episode + 1}/{episodes}")
            
        # --- YOUR CODE HERE --- #
        
    return rewards_per_episode, agent.q_table

def plot_rewards(rewards_dict):
    """
    Plots the sum of rewards per episode for each agent.
    """
    # --- YOUR CODE HERE --- #

def plot_policy(q_table, env, agent_name):
    """
    Visualizes the learned policy from a Q-table.
    """
    policy = np.argmax(q_table, axis=1)
    # Action mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    action_arrows = ['↑', '→', '↓', '←']
    
    # --- YOUR CODE HERE --- #
    
    print(f"\nFinal Policy for {agent_name}:")

# --- Main script execution ---
if __name__ == "__main__":
    # Use the render_mode="rgb_array" to avoid screen pop-ups
    env = gym.make("CliffWalking-v1")
    
    # --- SARSA Agent ---
    print("--- Training SARSA Agent ---")
    sarsa_agent = SarsaAgent(env.action_space.n, env.observation_space.n)
    sarsa_rewards, sarsa_q_table = train(sarsa_agent, env)
    
    # --- Q-Learning Agent ---
    print("\n--- Training Q-Learning Agent ---")
    q_learning_agent = QLearningAgent(env.action_space.n, env.observation_space.n)
    q_rewards, q_q_table = train(q_learning_agent, env)
    
    env.close()

    # --- Plotting and Analysis ---
    plot_rewards({
        "SARSA": sarsa_rewards,
        "Q-Learning": q_rewards
    })
    
    plot_policy(sarsa_q_table, env, "SARSA")
    plot_policy(q_q_table, env, "Q-Learning")