import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, alpha=0.5, gamma=1.0, epsilonStart=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilonStart
        self.qValues = np.zeros((env.observation_space.n, env.action_space.n))

    def selectAction(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.qValues[state])

    def update(self, state, action, reward, next_state, next_action, terminated):
        raise NotImplementedError

class SarsaAgent(Agent):
    def update(self, state, action, reward, next_state, next_action, terminated):
        target = reward if terminated else reward + self.gamma * self.qValues[next_state, next_action]
        self.qValues[state, action] += self.alpha * (target - self.qValues[state, action])

class SarsaMaxAgent(Agent):
    def update(self, state, action, reward, next_state, next_action, terminated):
        target = reward if terminated else reward + self.gamma * np.max(self.qValues[next_state])
        self.qValues[state, action] += self.alpha * (target - self.qValues[state, action])

def train(agent, numEpisodes=500, maxSteps=1000):
    episodeRewards = []
    for episode in range(numEpisodes):
        observation, _ = agent.env.reset()
        terminated = False
        totalReward = 0
        steps = 0

        action = agent.selectAction(observation)

        while not terminated and steps < maxSteps:
            prev_observation = observation
            observation, reward, terminated, _, _ = agent.env.step(action)
            totalReward += reward

            next_action = agent.selectAction(observation)

            agent.update(prev_observation, action, reward, observation, next_action, terminated)

            action = next_action
            steps += 1

        episodeRewards.append(totalReward)
        agent.epsilon = 1.0 / (episode + 1)

    return episodeRewards

def visualizePolicy(qValues, height=4, width=12, title='Policy', filename=None):
    policy = np.argmax(qValues, axis=1)
    actions = {0: '^', 1: '>', 2: 'v', 3: '<'}

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True)
    # Fixed y-labels: row 3 (bottom) to row 0 (top)
    ax.set_yticklabels([3, 2, 1, 0, ''])

    for r in range(height):
        for c in range(width):
            s = r * width + c
            if r == 3 and c == 0: label = 'S'
            elif r == 3 and c == 11: label = 'G'
            elif r == 3 and 0 < c < 11: label = 'C'
            else: label = actions.get(policy[s], '?')
            ax.text(c + 0.5, height - r - 0.5, label, ha='center', va='center')

    ax.set_title(title)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

# Main script
env = gym.make('CliffWalking-v1')

sarsaAgent = SarsaAgent(env)
sarsaMaxAgent = SarsaMaxAgent(env)

sarsaRewards = train(sarsaAgent)
sarsaMaxRewards = train(sarsaMaxAgent)

# Plot both on the same graph
plt.plot(sarsaRewards, label='SARSA')
plt.plot(sarsaMaxRewards, label='SARSAMAX')
plt.title('Sum of Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.ylim(-250, 0)
plt.legend()
plt.savefig('rewardsComparison.png')
plt.close()

# Visualize policies
visualizePolicy(sarsaAgent.qValues, title='SARSA Policy', filename='sarsaGrid.png')
visualizePolicy(sarsaMaxAgent.qValues, title='SARSAMAX Policy', filename='sarsaMaxGrid.png')

# Test both (greedy)
for name, agent in [('SARSA', sarsaAgent), ('SARSAMAX', sarsaMaxAgent)]:
    test_env = gym.make("CliffWalking-v1", render_mode="human")
    observation, _ = test_env.reset()
    terminated = False
    print(f"Testing {name}...")
    while not terminated:
        action = np.argmax(agent.qValues[observation])
        observation, _, terminated, _, _ = test_env.step(action)
    test_env.render()
    test_env.close()