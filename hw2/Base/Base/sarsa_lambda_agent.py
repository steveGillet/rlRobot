# sarsa_lambda_agent.py
import numpy as np

class SarsaLambdaAgent:
    """
    An agent that uses SARSA(λ) with accumulating eligibility traces.
    """
    def __init__(self, action_space_size, state_space_size, alpha=0.1, gamma=0.9, epsilon=0.1, lambda_val=0.9):
        self.q_table = np.zeros((state_space_size, action_space_size))
        # Eligibility trace table, initialized to zeros
        self.e_table = np.zeros((state_space_size, action_space_size))
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size

    def choose_action(self, state):
        """
        Action selection using epsilon-greedy policy.
        """
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            qVals = self.q_table[state]
            max = np.max(qVals)
            bestActions = np.where(qVals == max)[0]
            return np.random.choice(bestActions)

    def reset_traces(self):
        """
        Resets the eligibility traces to zero. Should be called at the start of each episode.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.e_table = np.zeros((self.state_space_size, self.action_space_size))

    def update(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-table and eligibility traces using the SARSA(λ) backward view.
        """
        delta = reward + self.gamma*self.q_table[next_state, next_action] - self.q_table[next_state, next_action]
        self.e_table[state, action] += 1
        self.q_table += self.alpha * delta * self.e_table
        self.e_table = self.gamma * self.lambda_val * self.e_table

