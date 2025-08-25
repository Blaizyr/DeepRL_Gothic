# utils/input_control.py
import random
import time
import pydirectinput as pdi

pdi.PAUSE = 0.01


class InputController:
    def __init__(self, action_map, default_hold_ms: int = 60):
        self.default_hold_ms = default_hold_ms
        self.action_map = action_map

    def execute_action(self, action_id: int, hold_ms: int = None):
        hold_ms = hold_ms or self.default_hold_ms
        action = self.action_map.get(action_id)
        timing_info = {}
        start = time.time()

        if action is None:
            time.sleep(hold_ms / 1000.0)
            return {'type': 'idle'}

        if isinstance(action, tuple):
            action_type = action[0]

            if action_type == 'jump':
                a_type, k1, k2, a_time_ms = action
                pdi.keyDown(k1)
                time.sleep(a_time_ms)
                pdi.keyDown(k2)
                time.sleep(0.02)
                pdi.keyUp(k2)
                time.sleep(max(0, a_time_ms))
                pdi.keyUp(k1)
                timing_info.update({'type': 'jump', 'keys': (k1, k2)})

            elif action_type == 'macro':
                a_type, mod, key, a_time_ms = action
                pdi.keyDown(mod)
                time.sleep(0.02)
                pdi.keyDown(key)
                time.sleep(0.02)
                pdi.keyUp(key)
                time.sleep(random.randrange(0, a_time_ms / 1000.0 - 0.04))
                pdi.keyUp(mod)
                timing_info.update({'type': 'macro', 'keys': (mod, key)})

            elif action_type == 'side_step':
                a_type, k1, k2, a_time_ms = action
                pdi.keyDown(k1)
                time.sleep(0.02)
                pdi.keyDown(k2)
                time.sleep(a_time_ms / 1000.0)
                pdi.keyUp(k2)
                time.sleep(0.02)
                pdi.keyUp(k1)
                timing_info.update({'type': 'side_step', 'keys': (k1, k2)})

            elif action_type == 'hold':
                a_type, key, a_time_ms = action
                actual_hold = random.randrange(hold_ms, a_time_ms) / 1000.0
                pdi.keyDown(key)
                time.sleep(actual_hold)
                pdi.keyUp(key)
                timing_info.update({'type': 'hold', 'key': key})

            elif action_type == 'bow_shooting':
                a_type, aim, shoot, a_time_ms = action
                actual_hold = random.randrange(1, a_time_ms / 1000.0)
                pdi.keyDown(aim)
                time.sleep(0.02)
                pdi.keyDown(shoot)
                time.sleep(actual_hold)
                pdi.keyUp(shoot)
                time.sleep(0.02)
                pdi.keyUp(aim)
                timing_info.update({'type': 'bow_shooting', 'keys': (aim, shoot)})

            else:
                timing_info.update({'type': 'unknown', 'action': action})

        else:
            key = action
            pdi.keyDown(key)
            time.sleep(hold_ms / 1000.0)
            pdi.keyUp(key)
            timing_info.update({'type': 'basic', 'key': key})

        timing_info['duration'] = time.time() - start
        return timing_info
