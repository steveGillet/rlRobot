# sarsa_agent.py
import numpy as np

class SarsaAgent:
    """
    An agent that learns to solve the Cliff Walking environment using the SARSA algorithm.
    """
    def __init__(self, action_space_size, state_space_size, alpha=0.5, gamma=0.9, epsilon=0.1):
        """
        Initializes the agent's Q-table and parameters.
        
        Args:
            action_space_size (int): The number of possible actions.
            state_space_size (int): The number of possible states.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
        """
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def choose_action(self, state):
        """
        Chooses an action from a given state using an epsilon-greedy policy.
        With probability epsilon, it chooses a random action (exploration).
        Otherwise, it chooses the action with the highest Q-value (exploitation).
        """
        # --- YOUR CODE HERE --- #

    def update(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-table using the SARSA update rule. This is an on-policy method.
        The update rule is:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        
        Args:
            state (int): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (int): The state transitioned to.
            next_action (int): The action that will be taken in the next state.
        """
        # --- YOUR CODE HERE --- #
