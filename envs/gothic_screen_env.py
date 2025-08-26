# envs/gothic_screen_env.py
import time
import cv2
import gymnasium as gym
import keyboard
import pytesseract
from gymnasium import spaces
import numpy as np

from utils.build_action_map import build_action_map
from utils.input_control import InputController
from utils.text_targeting import detect_text_candidates, candidates_to_targets, ocr_on_candidate, TargetHistory
from utils.screen_capture import ScreenCapture


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

        self.screen_capture = ScreenCapture(
            window_title_substr=self.window_title_substr,
            frame_shape=self.frame_shape,
        )

        # --- state ---
        self.prev_enemy_hp = None
        self.prev_player_hp = None
        self.prev_player_mana = None
        self.prev_player_oxygen = None
        self.prev_weapon_equipped = False
        self.weapon_equipped = False
        self.combat_started = False
        self.combat_timer = 0
        self.player_in_danger = False
        self.steps = 0

        # --- spaces ---
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_shape[0], frame_shape[1], 1), dtype=np.uint8
        )
        self.action_map, self.action_groups = build_action_map()
        self.input_controller = InputController(action_map=self.action_map, default_hold_ms=action_hold_ms)
        print(f"mapa akcji: {self.action_map}")
        print(f"grupy akcji: {self.action_groups}")

        self.action_space = spaces.Discrete(len(self.action_map))

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(0.2)
        self.steps = 0
        obs = self.screen_capture.get_observation(preprocess=True)
        self.steps = 0
        return obs, {}

    def step(self, action):
        action_id = int(action)
        timing_info = self.input_controller.execute_action(action_id, hold_ms=self.action_hold_ms)
        time.sleep(0.05)

        raw_obs = self.screen_capture.get_observation(preprocess=False)
        obs = self.screen_capture.get_observation(preprocess=True)

        reward = 0.0
        info = {}

        targets_boxes = detect_text_candidates(obs, limit=25)
        targets = candidates_to_targets(raw_obs, targets_boxes)

        best = None
        for cand in targets[:6]:
            cand = ocr_on_candidate(raw_obs, cand, refine=True)
            if cand.text and cand.conf and cand.conf >= 40:
                best = cand
                break

        if best:
            self.target_hist.push(best)
            dx, dy = best.offset

            tau = 420.0
            reward += 0.9 * np.exp(-abs(dx) / tau)

            if self.target_hist.stable_centered(dx_thresh=35, min_frames=5):
                reward += 10.0

            info["target_name"] = best.text
            info["target_dx"] = dx
            info["target_zone"] = best.zone
        else:
            self.target_hist = TargetHistory(maxlen=15)

        # --- baseline ---
        diff = float(np.mean(np.abs(obs.astype(np.int16) - self.prev_obs.astype(np.int16))))
        if diff >= self.stagnation_threshold:
            reward += 0.005
        else:
            reward -= 0.001
        info["frame_diff"] = diff

        # --- events ---
        enemy_hp, player_hp = detect_enemy_hp(raw_obs), detect_player_hp(raw_obs)
        player_mana, player_oxygen = detect_player_mana(raw_obs), detect_player_oxygen(raw_obs)
        xp_gain = detect_xp_gain(raw_obs) if self.steps % 10 == 0 else None

        if enemy_hp is not None and self.prev_enemy_hp is None:
            reward += 25
            info["event"] = "enemy_detected"

        if enemy_hp is not None and self.prev_enemy_hp is not None:
            if enemy_hp < self.prev_enemy_hp:
                if not self.combat_started:
                    self.combat_started = True
                    self.combat_timer = time.time()
                reward += -0.5 * (time.time() - self.combat_timer)
                info["event"] = "enemy_damaged"

        if player_hp is not None and self.prev_player_hp is not None:
            if player_hp < self.prev_player_hp:
                reward -= 5.0
                self.player_in_danger = True
                self.player_in_danger_timer = time.time()
                info["event"] = "player_damaged"

        if player_hp is not None:
            if player_hp == 0:
                reward -= 300.0
                info["event"] = "player_dead"
                terminated = True
            elif 1 <= player_hp <= 2:
                reward -= 100.0
                info["event"] = "player_defeated"
                terminated = True
            else:
                terminated = False
        else:
            terminated = False

        if enemy_hp is not None and enemy_hp <= 1:
            if xp_gain:
                reward += 125.0
                info["event"] = "enemy_killed"
                terminated = True
            else:
                info["pending_event"] = "enemy_maybe_killed"

        if player_oxygen is not None:
            if self.prev_player_oxygen is None:
                reward -= 100.0
                info["event"] = "player_under_water"

            elif player_oxygen < self.prev_player_oxygen:
                base_penalty = 10.0
                severity_multiplier = 1.0 + 2.0 * (1.0 - player_oxygen)  # np. 1.0 do 3.0
                reward -= base_penalty * severity_multiplier
                info["event"] = "oxygen_decreasing"

            if player_oxygen == 0:
                self.player_is_drowning = True
                reward -= 150.0
                info["event"] = "player_drowning"
                terminated = True

            else:
                self.player_is_drowning = False
        else:
            self.player_is_drowning = False
            if self.prev_player_oxygen is not None:
                reward += 50.0
                info["event"] = "exited_water"

        self.prev_player_oxygen = player_oxygen

        if player_mana is not None:
            self.weapon_equipped = "magic"
            reward += 0.5
            info["event"] = "player_got_some_magic"
        # Co robić jak wykryje magię?

        # --- macros reward ---
        if action_id in self.action_groups['macro']:
            macro_reward = self.compute_macro_dexterity_reward(timing_info)
            reward += macro_reward
            info["timing_reward"] = macro_reward

        if self.combat_started:
            if enemy_hp is not None and self.prev_enemy_hp is not None:
                if enemy_hp < self.prev_enemy_hp:
                    self.combat_timer = time.time()
            if time.time() - self.combat_timer > 10:
                reward -= 15
                self.combat_started = False
                info["event"] = "combat_timeout"

        # --- manual reward ---
        if self.manual_reward != 0.0:
            reward += self.manual_reward
            info["manual_reward"] = self.manual_reward
            self.manual_reward = 0.0

        # --- weapon change reward ---
        if action_id in self.action_groups['weapon']:
            if self.combat_started:
                reward -= 3
            reward += 0.05
            info["exploration_bonus"] = "weapon_slot"

        # --- update ---
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


def detect_player_oxygen(obs):
    return detect_bar_value(obs, roi=(1685, 2084, 2150, 2126), channel="blue")


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
