# training/train_gothic_cnn_ppo.py
import os
import time

import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gothic_screen_env import GothicScreenEnv

def make_env():
    return GothicScreenEnv(
        window_title_substr="GOTHIC 1.08k mod",
        frame_shape=(84, 84),
        max_steps=300,
        crop=None,
        action_hold_ms=60,
        stagnation_threshold=2.0
    )

def main():
    env = DummyVecEnv([make_env])
    model_path = "models/gothic_ppo_cnn.zip"

    def emergency_save(e=None):
        print("!! Emergency save triggered !!")
        os.makedirs("models", exist_ok=True)
        model.save("models/gothic_ppo_cnn_autosave.zip")
        env.close()

    keyboard.add_hotkey("backspace", emergency_save)

    if os.path.exists(model_path):
        print(">> Ładuję istniejący model…")
        model = PPO.load(model_path, env=env, device="cpu")
        model.set_env(env)
        reset_flag = False
    else:
        print(">> Tworzę nowy model…")
        model = PPO("CnnPolicy", env, verbose=1)
        reset_flag = True

    try:
        model.learn(total_timesteps=200_000, reset_num_timesteps=reset_flag)
    except KeyboardInterrupt:
        os.makedirs("models", exist_ok=True)
        model.save(model_path)

    env.close()



if __name__ == "__main__":
    time.sleep(10)
    main()
