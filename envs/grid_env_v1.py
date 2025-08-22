import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnvV1(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, size=(5,5), max_steps=50):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.observation_space = spaces.Box(low=0, high=max(size), shape=(2,), dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.array([0,0])
        self.goal_pos = np.array([self.size[0]-1, self.size[1]-1])
        self.steps = 0
        return self.agent_pos, {}

    def step(self, action):
        if action == 0 and self.agent_pos[1]<self.size[1]-1:  # UP
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1]>0:  # DOWN
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0]>0:  # LEFT
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0]<self.size[0]-1:  # RIGHT
            self.agent_pos[0] += 1

        self.steps += 1
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.steps>=self.max_steps
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.1
        return self.agent_pos, reward, done, False, {}

    def render(self):
        grid = np.full(self.size, '.')
        grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
        print("\n".join("".join(row) for row in grid))
        print()
