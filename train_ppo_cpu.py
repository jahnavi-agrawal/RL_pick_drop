"""
CPU PPO training for the force-aware Panda lift task.

Run from the project root:
python train_ppo_cpu.py --xml mujoco_menagerie/franka_emika_panda/panda_pick_place.xml

Requires pregrasp_q.npy in the project root because PandaLiftEnv.reset()
loads that file using a relative path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from pick_lift_env import PandaLiftEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="runs/panda_lift_baseline")
    parser.add_argument("--logdir", default="logs/panda_lift")
    return parser.parse_args()


def main():
    args = parse_args()

    xml_path = str(Path(args.xml).resolve())
    out_dir = Path(args.out)
    log_dir = Path(args.logdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not Path("pregrasp_q.npy").exists():
        raise FileNotFoundError(
            "pregrasp_q.npy was not found in the project root."
        )

    # Keep the first experiment CPU-friendly and reproducible.
    env = make_vec_env(
        PandaLiftEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "train"),
    )

    eval_env = make_vec_env(
        PandaLiftEnv,
        n_envs=1,
        seed=args.seed + 1000,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "eval"),
    )

    # Evaluate every ~5,000 training timesteps.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "evaluations"),
        eval_freq=max(5_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_panda",
    )

    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "xml_path": xml_path,
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "device": "cpu",
        "start_state": "pregrasp_q.npy",
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device="cpu",
            seed=args.seed,
        )

        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            tb_log_name="PPO_baseline",
        )

        model.save(str(out_dir / "final_model"))
        print(f"Final model saved to: {out_dir / 'final_model.zip'}")
        print(f"Best evaluated model: {out_dir / 'best' / 'best_model.zip'}")
        print(f"Configuration saved to: {out_dir / 'config.json'}")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
