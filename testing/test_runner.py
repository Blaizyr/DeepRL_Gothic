# training/test_runner.py
import yaml
import numpy as np
import matplotlib.pyplot as plt
from envs import grid_env_v1, grid_env_v2
from agents import ppo_agent
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_SCENARIOS_DIR = BASE_DIR / "testing/scenarios"

ENV_MAP = {
    "GridEnvV1": grid_env_v1.GridEnvV1,
    "GridEnvV2": grid_env_v2.GridEnvV2,
}

AGENT_MAP = {
    "PPO": ppo_agent.PPOAgent,
}


class TestRunner:
    def __init__(self, scenarios_file, model_dir=DEFAULT_MODELS_DIR, scenarios_dir=DEFAULT_SCENARIOS_DIR):
        self.scenarios_file = Path(scenarios_dir) / scenarios_file
        self.model_dir = Path(model_dir)

    def run(self):
        print("### Using scenarios:", self.scenarios_file)
        print("### Using model dir:", self.model_dir)
        with open(self.scenarios_file, "r") as f:
            scenarios = yaml.safe_load(f)

        for scen in scenarios:
            print(f"== Test scenario: {scen['name']} ==")
            env_class = ENV_MAP[scen["env_class"]]
            env = env_class(**scen.get("env_params", {}))

            agent_class = AGENT_MAP.get(scen.get("agent", "PPO"))
            agent = agent_class(env, **scen.get("agent_params", {}))
            agent.load(f"{self.model_dir}/{scen['name']}_{agent_class.__name__}")

            traj, heatmap = self.evaluate_agent(agent, env)
            self.save_results(scen['name'], traj, heatmap)
            env.close()

    def evaluate_agent(self, agent, env):
        obs = env.reset()[0]
        done = False
        traj = []
        heatmap = np.zeros(env.size if hasattr(env, "size") else (20, 20))

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            obs_int = np.array([int(obs[0]), int(obs[1])])
            traj.append([obs_int[0], obs_int[1], action, reward])

            if hasattr(env, "size"):
                pos = (obs_int[0], obs_int[1])
                heatmap[pos[1], pos[0]] += 1

        return np.array(traj), heatmap

    def save_results(self, scenario_name, traj, heatmap):
        np.save(f"{scenario_name}_traj.npy", traj)
        np.save(f"{scenario_name}_heatmap.npy", heatmap)

        plt.imshow(heatmap, cmap="hot", interpolation="nearest", origin="lower")
        plt.colorbar(label="Visits")

        traj_positions = traj[:, :2].astype(int)

        plt.plot(
            traj_positions[:, 0],
            traj_positions[:, 1],
            color="cyan",
            marker="o",
            markersize=3,
            linewidth=1,
            label="Trajectory"
        )

        plt.scatter(traj_positions[0, 0], traj_positions[0, 1], c="green", s=80, label="Start")
        plt.scatter(traj_positions[-1, 0], traj_positions[-1, 1], c="red", s=80, label="End")

        plt.title(f"Heatmap & Trajectory: {scenario_name}")
        plt.legend()
        plt.savefig(f"{scenario_name}_heatmap.png")
        plt.close()
