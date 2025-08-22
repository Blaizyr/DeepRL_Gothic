# generators/scenario_generator.py
import os
import yaml
from datetime import datetime


class BaseWorldGenerator:
    def __init__(self, output_dir="artifacts/"):
        self.output_dir = output_dir

    def generate(self, **kwargs):
        raise NotImplementedError


class GridWorldGenerator(BaseWorldGenerator):

    def generate(self, sizes=(2, 5, 10), num_maps=3, obstacles=True, agent="PPO", training_purpose=False):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(base_dir, exist_ok=True)

        scenarios = []
        for size, s in sizes:
            for i in range(num_maps):
                idx = s + i
                scenario = {
                    "name": f"{idx}_grid_{size}x{size}_map{i}",
                    "env_class": "GridEnvV2",
                    "env_params": {
                        "size": [size, size],
                        "obstacles": obstacles,
                        "max_steps": size * size * 2,

                    },
                    "seed": i,
                    "agent": agent,
                    "agent_params": {"policy_kwargs": None}
                }
                scenarios.append(scenario)

                path = os.path.join(base_dir, f"{scenario['name']}.yaml")
                with open(path, "w") as f:
                    yaml.dump(scenario, f)

        return scenarios, base_dir


if __name__ == "__main__":
    training = "../training/scenarios"
    test = "../testing/scenarios"
    generator = GridWorldGenerator(output_dir=test)
    generator.generate(sizes=(5, 7, 13, 25))
