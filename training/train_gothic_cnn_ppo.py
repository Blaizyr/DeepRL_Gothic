import os
import time
import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
from envs.gothic_screen_env import GothicScreenEnv


def make_env():
    return GothicScreenEnv(
        window_title_substr="GOTHIC 1.08k mod",
        frame_shape=(84, 84),
        max_steps=300,
        crop=None,
        action_hold_ms=60,
        stagnation_threshold=2.0
    )


class PPOModelCNNPolicyTrainer:
    def __init__(self, model_path="models/gothic_ppo_cnn.zip"):
        self.model_path = model_path
        self.env = DummyVecEnv([make_env])
        self.model, self.reset_flag = self._load_or_create_model()

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            print(">> Ładuję istniejący model…")
            model = PPO.load(self.model_path, env=self.env, device="cpu")
            model.set_env(self.env)
            reset_flag = False
        else:
            print(">> Tworzę nowy model…")
            model = PPO("CnnPolicy", self.env, verbose=1, device="cpu")
            reset_flag = True
        return model, reset_flag

    def _generate_save_path(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gothic_ppo_cnn_{timestamp}.zip"
        return os.path.join("models", filename)

    def manual_save(self):
        print(">> Ręczny zapis modelu z sygnaturą czasową...")
        os.makedirs("models", exist_ok=True)
        save_path = self._generate_save_path()
        self.model.save(save_path)
        print(f"Model zapisany jako: {save_path}")

    def emergency_shutdown(self):
        print(">> Aktywowano bezpieczne zamknięcie (F1)...")
        self.manual_save()
        os._exit(0)

    def hard_shutdown(self):
        print(">> Wymuszono przerwanie (F4) — postęp NIE zostanie zapisany!")
        os._exit(1)

    def run(self):
        print("Naciśnij F1, aby zapisać i bezpiecznie zakończyć.")
        print("Naciśnij F4, aby przerwać sesję bez zapisu.")

        keyboard.add_hotkey("F1", self.emergency_shutdown)
        keyboard.add_hotkey("F4", self.hard_shutdown)

        try:
            self.model.learn(total_timesteps=200_000, reset_num_timesteps=self.reset_flag)
            self.reset_flag = False
        except Exception as e:
            print(f"Wystąpił błąd podczas uczenia: {e}")
        finally:
            os.makedirs("models", exist_ok=True)
            self.model.save(self.model_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Zapisano postęp: {self.model_path} - {timestamp}")
            print("Zamykanie środowiska...")
            self.env.close()
            print("Środowisko zamknięte. Koniec działania programu.")


if __name__ == "__main__":
    trainer = PPOModelCNNPolicyTrainer()
    for i in range(10, 0, -1):
        print(f"Model aktywuje się za: {i}")
        time.sleep(1)
    trainer.run()
