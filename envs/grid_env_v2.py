import numpy as np
from .grid_env_v1 import GridEnvV1

class GridEnvV2(GridEnvV1):
    def __init__(self, size=(5,5), obstacles=True, max_steps=50):
        super().__init__(size, max_steps)
        self.obstacles = obstacles
        self.grid = np.zeros(size, dtype=int)
        if self.obstacles:
            self.grid[1,1] = 1
            self.grid[2,3] = 1

    def step(self, action):
        new_pos = self.agent_pos.copy()
        if action == 0 and new_pos[1]<self.size[1]-1:  # UP
            new_pos[1] += 1
        elif action == 1 and new_pos[1]>0:  # DOWN
            new_pos[1] -= 1
        elif action == 2 and new_pos[0]>0:  # LEFT
            new_pos[0] -= 1
        elif action == 3 and new_pos[0]<self.size[0]-1:  # RIGHT
            new_pos[0] += 1

        if self.obstacles and self.grid[new_pos[1], new_pos[0]] == 1:
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        self.steps += 1
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.steps>=self.max_steps
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.1
        return self.agent_pos, reward, done, False, {}
