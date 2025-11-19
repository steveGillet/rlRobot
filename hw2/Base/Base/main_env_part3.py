# maze_env.py

import gymnasium as gym
import numpy as np

class MazeEnv(gym.Env):
    """A simple 5x5 maze environment with a delayed reward."""
    def __init__(self, render_mode=None):
        super(MazeEnv, self).__init__()
        self.size = 5
        self.action_space = gym.spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = gym.spaces.Discrete(self.size * self.size)
        
        # Define the maze layout
        # 1 = Wall, 0 = Path, 2 = Start, 3 = Goal
        self.maze = np.array([
            [2, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 3]
        ])
        
        self.start_pos = np.argwhere(self.maze == 2)[0]
        self.goal_pos = np.argwhere(self.maze == 3)[0]
        self.agent_pos = None

    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.copy(self.start_pos)
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action):
        new_pos = np.copy(self.agent_pos)
        if action == 0:  # Up
            new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Down
            new_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3:  # Left
            new_pos[1] = max(0, self.agent_pos[1] - 1)
            
        # Check for walls
        if self.maze[new_pos[0], new_pos[1]] != 1:
            self.agent_pos = new_pos

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        reward = 10.0 if terminated else 0.0
        
        return self._pos_to_state(self.agent_pos), reward, terminated, False, {}

    def render(self):
        # A simple console renderer
        grid = self.maze.astype(str)
        grid[grid == '0'] = '.'
        grid[grid == '1'] = '█'
        grid[grid == '2'] = 'S'
        grid[grid == '3'] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print("\n".join(" ".join(row) for row in grid))
        print("-" * 10)