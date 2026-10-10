import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from pick_place_env import PandaPickPlaceEnv

def main():
    env = PandaPickPlaceEnv(randomize_cube=False)
    env = Monitor(env, filename="logs/monitor.csv")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path="models/",
        name_prefix="panda_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log="logs/tensorboard/",
        device="cpu",
    )

    print("\nStarting PPO training. The initial policy is random.")
    print("Training is headless; use visualize_ppo.py to watch saved policies.")
    model.learn(total_timesteps=300_000, callback=checkpoint)
    model.save("models/panda_ppo_final")
    env.close()
    print("Saved models/panda_ppo_final.zip")

if __name__ == "__main__":
    main()
