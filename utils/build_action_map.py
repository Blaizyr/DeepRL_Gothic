from itertools import count


def build_action_map():
    action_map = {}
    action_groups = {
        'basic': [],
        'modifier': [],
        'weapon': [],
        'macro': [],
        'jump': [],
        'side_step': [],
        'hold': [],
        'bow_shooting': [],
    }
    action_id = count(0)

    # === 1. Podstawowe ruchy ===
    for base in [None, "up", "down", "left", "right", "delete", "page_down"]:
        aid = next(action_id)
        action_map[aid] = base
        action_groups['basic'].append(aid)

    # === 2. Akcje interakcji ===
    for mod in ["ctrl", "alt"]:
        aid = next(action_id)
        action_map[aid] = mod
        action_groups['modifier'].append(aid)

    # === 3. Wybór broni ===
    for num in ["1", "2", "3"]:
        aid = next(action_id)
        action_map[aid] = num
        action_groups['weapon'].append(aid)

    # === 4. Makra: Ctrl + strzałka ===
    for direction in ["right", "left", "up", "down"]:
        aid = next(action_id)
        action_map[aid] = ('macro', 'ctrl', direction, 100)
        action_groups['macro'].append(aid)

    # === 5. Makro skoków  ===

     # aid = next(action_id)
     # action_map[aid] = ('jump', 'alt', 'up', 100) climb
     # action_groups['jump'].append(aid)

    aid = next(action_id)
    action_map[aid] = ('jump', 'up', 'alt', 500)
    action_groups['jump'].append(aid)

     # action_map[next(action_id)] = ('macro', 'up', 'num 0', 300)  # up + num 0

    # === 6. Side stepy ===
    for keys in [('right', 'del'), ('left', 'pgdown')]:
        aid = next(action_id)
        action_map[aid] = ('side_step', *keys, 250)
        action_groups['side_step'].append(aid)

    # === 7. Długie trzymanie klawiszy (hold) ===
    aid = next(action_id)
    action_map[aid] = ('hold', 'up', 2500)
    action_groups['hold'].append(aid)

    # === 8. Strzelanie z łuku ===
    aid = next(action_id)
    action_map[aid] = ('bow_shooting', 'ctrl', 'up', 5000)
    action_groups['bow_shooting'].append(aid)

    return action_map, action_groups
