# training/runner.py
import os
import yaml
from pathlib import Path
from envs import grid_env_v1, grid_env_v2
from agents import ppo_agent

ENV_MAP = {
    "GridEnvV1": grid_env_v1.GridEnvV1,
    "GridEnvV2": grid_env_v2.GridEnvV2,
}

AGENT_MAP = {
    "PPO": ppo_agent.PPOAgent,
}


class TrainingRunner:
    def __init__(self, scenarios_dir="scenarios", artifacts_dir="artifacts"):
        self.scenarios_dir = scenarios_dir
        self.artifacts_dir = artifacts_dir

        """   
# def run_all_from_dir(self, stamp):
#        stamp_path = Path(self.scenarios_dir) / stamp
#        for fname in sorted(os.listdir(stamp_path)):
#            if fname.endswith(".yaml") or fname.endswith(".yml"):
#                fpath = os.path.join(stamp_path, fname)
#                self.run_all_from_file(fpath)
# 
#    def run_all_from_file(self, scenarios_file):
#        with open(scenarios_file, "r") as f:
#            data = yaml.safe_load(f)
# 
#        if isinstance(data, dict):
#            scenarios = [data]
#        elif isinstance(data, list):
#            scenarios = data
#        else:
#            raise ValueError(f"Unexpected YAML structure in {scenarios_file}: {type(data)}")
# 
#        for idx, scen in enumerate(scenarios):
#            self.run_single(scen, idx)
# 
#    def run_single(self, scen, idx):
#        env_class = ENV_MAP[scen["env_class"]]
#        env = env_class(**scen.get("env_params", {}))
# 
#        agent_class = AGENT_MAP[scen.get("agent", "PPO")]
#        agent = agent_class(env, **scen.get("agent_params", {}))
# 
#        print(f"[RUNNER] Training scenario {idx}: {scen['name']} with {agent_class.__name__}")
#        agent.train()
# 
#        out_name = f"{scen['name']}_{agent_class.__name__}.zip"
#        out_path = os.path.join(self.artifacts_dir, out_name)
#        os.makedirs(self.artifacts_dir, exist_ok=True)
#        agent.save(out_path)
# 
#        env.close()
        """

    def run_all_from_dir_as_sequence_for_agent(self, stamp):
        stamp_path = Path(self.scenarios_dir) / stamp
        all_scenarios = []

        for fname in sorted(os.listdir(stamp_path)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(stamp_path, fname)
                with open(fpath, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        all_scenarios.append(data)
                    elif isinstance(data, list):
                        all_scenarios.extend(data)

        self.run_sequence_for_agent(all_scenarios, stamp)

    def run_sequence_for_agent(self, scenarios, stamp):
        if not scenarios:
            print("[RUNNER] No scenarios found.")
            return

        agent_class = AGENT_MAP[scenarios[0].get("agent", "PPO")]
        first_env_class = ENV_MAP[scenarios[0]["env_class"]]
        env = first_env_class(**scenarios[0].get("env_params", {}))
        agent = agent_class(env, **scenarios[0].get("agent_params", {}))

        for idx, scen in enumerate(scenarios):
            env_class = ENV_MAP[scen["env_class"]]
            env = env_class(**scen.get("env_params", {}))
            agent.model.set_env(env)
            print(f"[RUNNER] Training scenario {idx}: {scen['name']}")
            agent.train()
            env.close()

        out_name = f"{stamp}_{agent_class.__name__}.zip"
        out_path = os.path.join(self.artifacts_dir, out_name)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        agent.save(out_path)
        print(f"[RUNNER] Saved trained model: {out_path}")


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.run_all_from_dir_as_sequence_for_agent(stamp="2025-08-22_10-43-15")  # copy dir timestamp
