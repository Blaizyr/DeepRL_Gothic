# utils/input_control.py
import time
import pydirectinput as pdi

pdi.PAUSE = 0.01

def send_action(action_id: int, hold_ms: int = 60):
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
        16: ('macro', 'up', 'num 0', 300),
        # 16: ('macro', 'down', 'num 0', 300),
        # 16: ('macro', 'ctrl', 'alt', 'left', 100),
        # 15: ('macro', 'ctrl', 'alt', 'right', 100),
    }

    action = action_map.get(action_id)
    timing_info = {}

    if action is None:
        time.sleep(hold_ms / 1000.0)
        timing_info['type'] = 'idle'
        return timing_info

    if action[0] != 'macro':
        key = action
        start = time.time()
        pdi.keyDown(key)
        time.sleep(3 * hold_ms / 1000.0)
        pdi.keyUp(key)
        end = time.time()
        timing_info['duration'] = end - start
        timing_info['type'] = 'basic'

    else:
        _, mod_key, key, _ = action
        start = time.time()

        pdi.keyDown(mod_key)
        time.sleep(0.02)
        pdi.keyDown(key)
        inner_delay = 0.02
        time.sleep(0.02)
        pdi.keyUp(key)
        time.sleep((hold_ms / 1000.0) - 0.04)
        pdi.keyUp(mod_key)

        total_time = time.time() - start
        timing_info.update({
            'type': 'macro',
            'action_to_arrow_delay': inner_delay,
            'total_duration': total_time,
            'requested_hold_ms': hold_ms
        })

    return timing_info
