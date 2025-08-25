# testing/rollout_gothic.py
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gothic_screen_env import GothicScreenEnv

def make_env():
    return GothicScreenEnv(
        window_title_substr="Gothic",
        frame_shape=(84,84),
        max_steps=500,
        stagnation_threshold=2.0
    )

def main():
    env = DummyVecEnv([make_env])
    model = PPO.load("../training/models/gothic_ppo_cnn", env=env)
    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if np.any(done):
            obs = env.reset()
        time.sleep(0.01)
    env.close()

if __name__ == "__main__":
    main()
