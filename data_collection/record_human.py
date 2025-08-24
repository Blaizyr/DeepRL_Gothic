# data_collection/record_human.py
import os
import time
from datetime import datetime

import numpy as np
import keyboard
from utils.window_capture import find_window_by_title_substring, grab_window
from utils.vision import preprocess_rgb_to_obs

action_map = {
    0: None,
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "ctrl",
    6: "alt",
    7: "1",
    8: "2",
    9: "3",
    10: ('macro', 'ctrl', 'right', 100),
    11: ('macro', 'ctrl', 'left', 100),
    12: ('macro', 'ctrl', 'up', 100),
    13: ('macro', 'ctrl', 'down', 100),
    14: ('macro', 'alt', 'up', 100),
    15: ('macro', 'up', 'alt', 300),
    16: ('macro', 'up', 'num 0', 300)
}

def main():
    os.makedirs("artifacts", exist_ok=True)
    hwnd = find_window_by_title_substring("Gothic")
    if hwnd is None:
        print("Nie znaleziono okna Gothic.")
        return

    frames = []
    actions = []

    print("Start za 3s… ustaw focus na Gothic.")
    time.sleep(8)

    try:
        while True:
            img = grab_window(hwnd)
            if img is None:
                continue
            obs = preprocess_rgb_to_obs(img, out_size=(84,84), gray=False)
            action_id = 0
            for k, aid in action_map.items():
                if keyboard.is_pressed(k):
                    action_id = aid
                    break
            frames.append(obs)
            actions.append(action_id)

            # ESC ends recording
            # if keyboard.is_pressed('esc'):

            # F1 ends recording
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
