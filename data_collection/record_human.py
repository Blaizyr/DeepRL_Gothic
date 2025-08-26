# data_collection/record_human.py
import os
import time
from datetime import datetime

import numpy as np
import keyboard

from utils.build_action_map import build_action_map
from utils.screen_capture import ScreenCapture

def main():
    os.makedirs("artifacts", exist_ok=True)
    screen_capture = ScreenCapture(window_title_substr="Gothic")
    hwnd = screen_capture.hwnd
    if hwnd is None:
        print("Nie znaleziono okna Gothic.")
        return

    frames = []
    actions = []

    print("Start za 3s… ustaw focus na Gothic.")
    time.sleep(8)

    action_map, _ = build_action_map()

    try:
        while True:
            obs = screen_capture.get_observation(preprocess = True)
            action_id = 0
            for k, aid in action_map.items():
                if keyboard.is_pressed(k):
                    action_id = aid
                    break
            frames.append(obs)
            actions.append(action_id)

            if keyboard.is_pressed('F1'):
                break

            time.sleep(0.03)  # ~33 FPS
    finally:
        frames = np.array(frames, dtype=np.uint8)         # (N,84,84,1)
        actions = np.array(actions, dtype=np.int64)       # (N,)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.savez_compressed(f"artifacts/{timestamp}_human_dataset.npz", obs=frames, act=actions)
        print(f"Zapisano artifacts/{timestamp}human_dataset.npz")

if __name__ == "__main__":
    main()
