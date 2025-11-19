import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from q_network import QNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size=147, action_size=3, seed=42, 
                 lr=1e-3, gamma=0.99, batch_size=64, 
                 buffer_size=100000, update_every=100, double_dqn=False, use_replay=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.double_dqn = double_dqn
        self.use_replay = use_replay
        self.timestep = 0

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return action_values.argmax(dim=1).cpu().item()
        else:
            return random.randrange(self.action_size)

    def step(self, state, action, reward, next_state, done):
        if self.use_replay:
            self.memory.push(state, action, reward, next_state, done)

            self.timestep += 1

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

        else:
            states = np.array([state])           
            actions = np.array([action])         
            rewards = np.array([reward])         
            next_states = np.array([next_state]) 
            dones = np.array([done], dtype=np.float32)  

            experiences = (states, actions, rewards, next_states, dones)
            self.learn(experiences)
            self.timestep += 1

        if self.timestep % self.update_every == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device).unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().to(device).unsqueeze(1)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device).unsqueeze(1)

        if self.double_dqn:
            next_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()