from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

class PPOAgent:
    def __init__(self, env, policy_kwargs=None, **kwargs):
        # Stable Baselines requires vector env
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", self.env, verbose=1, policy_kwargs=policy_kwargs, **kwargs)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load_from_path(self, path):
        self.model = PPO.load(path, env=self.env)
