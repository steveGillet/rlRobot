import gymnasium as gym
import numpy as np
env = gym.make('CliffWalking-v1')
observation, _ = env.reset()
qValues = np.zeros((env.observation_space.n, env.action_space.n))
episodes = 0
terminated = False
epsilon = 0.1
gammaPerm = 0.9
alpha = 0.5
maxSteps = 1000

while episodes < 500:
    steps = 0
    while terminated is False and steps < maxSteps:
        if np.random.rand() < epsilon:
            action = int(np.floor(np.random.rand() * 4))
        else:
            action = np.argmax(qValues[observation])

        previousObservation = observation
        observation, reward, terminated, _, _ = env.step(action)
        if np.random.rand() < epsilon:
            nextAction = int(np.floor(np.random.rand() * 4))
        else:
            nextAction = np.argmax(qValues[observation])

        gamma = 0 if terminated is True else gammaPerm
        qValues[previousObservation, action] =  qValues[previousObservation, action] + alpha * (reward + gamma * qValues[observation, nextAction] - qValues[previousObservation, action])
        
        steps += 1

    observation, _ = env.reset()
    episodes += 1
    terminated = False
    # epsilon = 1 / (episodes + 1)

testEnv = gym.make("CliffWalking-v1", render_mode="human")
observation, _ = testEnv.reset()

terminated = False
print(qValues)
while terminated is False:
    observation, _, terminated, _, _ = testEnv.step(np.argmax(qValues[observation]))


testEnv.render()