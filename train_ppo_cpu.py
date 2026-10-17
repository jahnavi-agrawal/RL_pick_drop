"""
CPU PPO training script for the force-aware Panda lift task.

Run from the project root:
  python train_ppo_cpu.py --xml mujoco_menagerie/franka_emika_panda/panda_pick_place.xml

This uses the initial/simple PPO setup (not the later 32-env, 3M-timestep script).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from pick_lift_env import PandaLiftEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml",
        help="Path to the MuJoCo scene XML."
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/panda_lift_ppo")
    parser.add_argument("--logdir", default="logs/panda_lift")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = str(Path(args.xml).resolve())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    def make_env():
        return Monitor(PandaLiftEnv(xml_path=xml_path))

    # CPU-only: explicitly set device="cpu"; no CUDA required.
    env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=args.logdir,
        vec_env_cls=None,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.logdir,
        device="cpu",
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.out)
    print(f"Saved PPO model to: {args.out}.zip")
    env.close()


if __name__ == "__main__":
    main()
