# envs/gothic_screen_env.py
import time
import cv2
import gymnasium as gym
import keyboard
import pytesseract
from gymnasium import spaces
import numpy as np
from utils.window_capture import find_window_by_title_substring, grab_window
from utils.input_control import send_action
from utils.vision import preprocess_rgb_to_obs
from utils.text_targeting import detect_text_candidates, candidates_to_targets, ocr_on_candidate, TargetHistory


class GothicScreenEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 window_title_substr="GOTHIC 1.08k mod",
                 frame_shape=(84, 84),
                 max_steps=300,
                 crop=None,
                 action_hold_ms=60,
                 stagnation_threshold=2.0):
        super().__init__()
        self.window_title_substr = window_title_substr
        self.frame_shape = frame_shape
        self.crop = crop
        self.max_steps = max_steps
        self.action_hold_ms = action_hold_ms
        self.stagnation_threshold = stagnation_threshold

        self.target_hist = TargetHistory(maxlen=15)
        self.debug_draw_text = False

        # --- state ---
        self.prev_obs = None
        self.prev_enemy_hp = None
        self.prev_player_hp = None
        self.prev_weapon_equipped = False
        self.combat_started = False
        self.combat_timer = 0
        self.steps = 0
        self.hwnd = None

        # --- spaces ---
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_shape[0], frame_shape[1], 1), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(17)

        # --- manual reward ---
        self.manual_reward = 0.0

        def reward_plus():
            self.manual_reward += 5.0
            print("[Manual Reward] +5")

        def reward_large_plus():
            self.manual_reward += 25.0
            print("[Manual Reward] +25")

        def reward_minus():
            self.manual_reward -= 5.0
            print("[Manual Reward] -5")

        def reward_large_minus():
            self.manual_reward -= 25.0
            print("[Manual Reward] -25")

        # Binds
        keyboard.add_hotkey("0", reward_large_minus)  ## -25
        keyboard.add_hotkey("-", reward_minus)  ######## -5
        keyboard.add_hotkey("=", reward_plus)  ######### +5
        keyboard.add_hotkey("\\", reward_large_plus)  ## +25

    def _capture_obs(self):
        if self.hwnd is None:
            self.hwnd = find_window_by_title_substring(self.window_title_substr)
            if self.hwnd is None:
                return None
        img = grab_window(self.hwnd, bbox=None)
        if img is None or getattr(img, "size", 0) == 0:
            print("[ERROR] grab_window failed or empty frame")
            return None
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(0.2)
        self.prev_obs = self._capture_obs()
        self.steps = 0
        if self.prev_obs is None:
            self.prev_obs = np.zeros((*self.frame_shape, 1), dtype=np.uint8)
        return self.prev_obs, {}

    def step(self, action):
        action_id = int(action)
        timing_info = send_action(action_id, hold_ms=self.action_hold_ms)
        time.sleep(0.05)

        raw_image = self._capture_obs()
        if raw_image is None or getattr(raw_image, "size", 0) == 0:
            print("[WARN] Got empty frame in step(), skipping detection")
            return self.prev_obs.copy(), 0.0, False, False, {}
        obs = preprocess_rgb_to_obs(raw_image, out_size=self.frame_shape, crop=self.crop, gray=True)

        reward = 0.0
        info = {}

        targets_boxes = detect_text_candidates(raw_image, limit=25)
        targets = candidates_to_targets(raw_image, targets_boxes)

        best = None
        for cand in targets[:6]:
            cand = ocr_on_candidate(raw_image, cand, refine=True)
            if cand.text and cand.conf and cand.conf >= 40:
                best = cand
                break

        if best:
            self.target_hist.push(best)
            dx, dy = best.offset

            tau = 420.0
            reward += 0.8 * np.exp(-abs(dx) / tau)

            if self.target_hist.stable_centered(dx_thresh=35, min_frames=5):
                reward += 1.0

            info["target_name"] = best.text
            info["target_dx"] = dx
            info["target_zone"] = best.zone
        else:
            self.target_hist = TargetHistory(maxlen=15)

        # --- baseline ---
        diff = float(np.mean(np.abs(obs.astype(np.int16) - self.prev_obs.astype(np.int16))))
        if diff >= self.stagnation_threshold:
            reward += 0.01
        else:
            reward -= 0.005
        info["frame_diff"] = diff

        # --- events ---
        enemy_hp, player_hp = detect_enemy_hp(raw_image), detect_player_hp(raw_image)
        xp_gain = detect_xp_gain(raw_image) if self.steps % 10 == 0 else None

        if enemy_hp is not None and self.prev_enemy_hp is None:
            reward += 0.25
            info["event"] = "enemy_detected"

        if enemy_hp is not None and self.prev_enemy_hp is not None:
            if enemy_hp < self.prev_enemy_hp:
                if not self.combat_started:
                    self.combat_started = True
                    self.combat_timer = time.time()
                reward += -0.1 * (time.time() - self.combat_timer)
                info["event"] = "enemy_damaged"

        if player_hp is not None and self.prev_player_hp is not None:
            if player_hp < self.prev_player_hp:
                reward -= 5.0
                info["event"] = "player_damaged"

        if player_hp is not None and player_hp <= 5:
            reward -= 100.0
            info["event"] = "player_dead"
            terminated = True
        else:
            terminated = False

        if enemy_hp is not None and enemy_hp <= 1:
            if xp_gain:
                reward += 125.0
                info["event"] = "enemy_killed"
                terminated = True
            else:
                info["pending_event"] = "enemy_maybe_killed"

        # --- macros reward ---
        if action_id in range(11, 16):
            macro_reward = self.compute_macro_dexterity_reward(timing_info)
            reward += macro_reward
            info["timing_reward"] = macro_reward

        if self.combat_started:
            if enemy_hp is not None and self.prev_enemy_hp is not None:
                if enemy_hp < self.prev_enemy_hp:
                    self.combat_timer = time.time()
            if time.time() - self.combat_timer > 10:
                reward -= 5
                self.combat_started = False
                info["event"] = "combat_timeout"

        # --- manual reward ---
        if self.manual_reward != 0.0:
            reward += self.manual_reward
            info["manual_reward"] = self.manual_reward
            self.manual_reward = 0.0

        # --- weapon change reward ---
        if action_id in range(8, 10):
            reward += 0.1
            info["exploration_bonus"] = "weapon_slot"

        # --- update ---
        self.prev_obs = obs
        self.prev_enemy_hp = enemy_hp
        self.prev_player_hp = player_hp
        self.steps += 1

        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, info

    def compute_macro_dexterity_reward(self, timing_info):
        base_reward = 0.0
        delay = timing_info['action_to_arrow_delay']
        if delay < 0.03:
            base_reward += 1.0
        elif delay < 0.1:
            base_reward += 0.5
        else:
            base_reward -= 0.5
        return base_reward

    def close(self):
        pass

# --- Detections ---
def detect_enemy_hp(obs):
    return detect_bar_value(obs, roi=(1685, 28, 2150, 70), channel="red")


def detect_player_hp(obs):
    return detect_bar_value(obs, roi=(40, 2084, 505, 2126), channel="red")


def detect_player_mana(obs):
    return detect_bar_value(obs, roi=(3325, 2080, 3795, 2126), channel="blue")


def detect_xp_gain(obs, roi=(1694, 1080, 2145, 1128)):
    if obs is not None:
        crop = obs[roi[0]:roi[1], roi[2]:roi[3], :]
        text = pytesseract.image_to_string(crop, lang="pol")
        return "Doświadczenie" in text
    return None


def detect_lvl_up(obs, roi=(1600, 1190, 2230, 1287)):
    crop = obs[roi[0]:roi[1], roi[2]:roi[3], :]
    text = pytesseract.image_to_string(crop, lang="pol")
    return "Kolejny Poziom" in text


def detect_bar_value(obs, roi, channel="red", threshold=100):
    y1, y2, x1, x2 = roi
    if obs is not None:
        crop = obs[y1:y2, x1:x2, :]
        ch_idx = {"red": 0, "green": 1, "blue": 2}[channel]
        channel_data = crop[:, :, ch_idx]
        active = (channel_data > threshold).astype(np.uint8)
        filled_ratio = active.sum() / active.size
        return float(filled_ratio)
    return None


def debug_show_roi(img, roi):
    y1, y2, x1, x2 = roi
    debug = img.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("ROI debug", debug)
    cv2.waitKey(0)
