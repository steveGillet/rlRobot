import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v1')
qValues = np.zeros((env.observation_space.n, env.action_space.n))
episodes = 0
epsilon = 1.0
gammaPerm = 1.0
alpha = 0.5
maxSteps = 1000
episodeRewards = []

while episodes < 500:
    steps = 0
    observation, _ = env.reset()
    terminated = False
    totalReward = 0

    if np.random.rand() < epsilon:
        action = int(np.floor(np.random.rand() * 4))
    else:
        action = np.argmax(qValues[observation])
        
    while terminated is False and steps < maxSteps:
        previousObservation = observation
        observation, reward, terminated, _, _ = env.step(action)
        totalReward += reward

        if np.random.rand() < epsilon:
            nextAction = int(np.floor(np.random.rand() * 4))
        else:
            nextAction = np.argmax(qValues[observation])

        gamma = 0 if terminated is True else gammaPerm
        qValues[previousObservation, action] =  qValues[previousObservation, action] + alpha * (reward + gamma * np.max(qValues[observation]) - qValues[previousObservation, action])
        
        action = nextAction
        steps += 1

    episodes += 1
    epsilon = 1 / (episodes + 1)
    episodeRewards.append(totalReward)

testEnv = gym.make("CliffWalking-v1", render_mode="human")
observation, _ = testEnv.reset()

terminated = False
print(qValues)
while not terminated:
    observation, _, terminated, _, _ = testEnv.step(np.argmax(qValues[observation]))

testEnv.render()

plt.plot(episodeRewards)
plt.title('Sum of Rewards per Episode (SARSAMAX)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('sarsaMaxRewards.png')
plt.close()

def visualize_policy(q_values, height=4, width=12, title='Policy', filename=None):
    policy = np.argmax(q_values, axis=1)
    actions = {0: '^', 1: '>', 2: 'v', 3: '<'}

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True)
    ax.set_yticklabels([4, 3, 2, 1, 0])

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

visualize_policy(qValues, filename="sarsaMaxGrid.png")