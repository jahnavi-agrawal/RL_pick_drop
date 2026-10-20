"""CPU PPO training and a small, sequential hyperparameter experiment.

Run one training job from the project root:
    python train_ppo_cpu.py --xml mujoco_menagerie/franka_emika_panda/panda_pick_place.xml

Compare a few resource-conscious PPO settings:
    python train_ppo_cpu.py --sweep

The sweep runs one environment at a time and writes each trial to its own
directory. It requires grasp_q.npy in the project root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from pick_lift_env import PandaLiftEnv


BASE_HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
}

# Change one setting at a time so the comparisons are interpretable.
SWEEP_TRIALS = [
    {"name": "baseline", "overrides": {}},
    {"name": "learning_rate_1e4", "overrides": {"learning_rate": 1e-4}},
    {"name": "entropy_003", "overrides": {"ent_coef": 0.003}},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out", default="runs/panda_grasp_lift_v1")
    parser.add_argument("--logdir", default="logs/panda_grasp_lift_v1")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run three sequential, one-setting-at-a-time PPO comparisons.",
    )
    parser.add_argument(
        "--sweep-timesteps",
        type=int,
        default=1500000,
        help="Timesteps per trial in --sweep mode (default: 150000).",
    )
    return parser.parse_args()


def train_trial(
    *,
    xml_path: str,
    seed: int,
    timesteps: int,
    n_envs: int,
    out_dir: Path,
    log_dir: Path,
    hyperparameters: dict,
    trial_name: str,
    sweep_mode: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        PandaLiftEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "train"),
    )
    eval_env = make_vec_env(
        PandaLiftEnv,
        n_envs=1,
        seed=seed + 1000,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "eval"),
    )

    if sweep_mode:
        # Keep evaluation overhead modest while applying the same protocol to
        # every candidate. No periodic model copies are needed for short trials.
        eval_freq = max(25_000 // n_envs, 1)
        n_eval_episodes = 5
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(out_dir / "best"),
                log_path=str(out_dir / "evaluations"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=0,
            )
        ]
    else:
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(out_dir / "best"),
                log_path=str(out_dir / "evaluations"),
                eval_freq=max(5_000 // n_envs, 1),
                n_eval_episodes=20,
                deterministic=True,
                render=False,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=max(10_000 // n_envs, 1),
                save_path=str(out_dir / "checkpoints"),
                name_prefix="ppo_panda",
            ),
        ]

    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "xml_path": xml_path,
        "total_timesteps": timesteps,
        "n_envs": n_envs,
        "seed": seed,
        **hyperparameters,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "start_state": "grasp_q.npy",
        "trial": trial_name,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    try:
        model = PPO(
            "MlpPolicy",
            env,
            **hyperparameters,
            verbose=1 if not sweep_mode else 0,
            tensorboard_log=str(log_dir / "tensorboard"),
            device="cpu",
            seed=seed,
        )
        eval_callback = callbacks[0]
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=not sweep_mode,
            tb_log_name=trial_name,
        )
        model.save(str(out_dir / "final_model"))

        result = {
            "trial": trial_name,
            "hyperparameters": hyperparameters,
            "timesteps": timesteps,
            "last_eval_mean_reward": eval_callback.last_mean_reward,
            "best_eval_mean_reward": eval_callback.best_mean_reward,
            "best_model": str(out_dir / "best" / "best_model.zip"),
            "final_model": str(out_dir / "final_model.zip"),
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"Finished {trial_name}: best eval reward = "
              f"{eval_callback.best_mean_reward:.3f}")
        return result
    finally:
        env.close()
        eval_env.close()


def main():
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1.")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1.")
    if args.sweep and args.sweep_timesteps < 25_000:
        raise ValueError("--sweep-timesteps must be at least 25000.")

    torch.set_num_threads(args.torch_threads)
    # The sweep is intentionally serial with one simulator process at a time.
    xml_path = str(Path(args.xml).resolve())
    if not Path(xml_path).is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    if not Path("grasp_q.npy").is_file():
        raise FileNotFoundError(
            "grasp_q.npy was not found in the project root. "
            "Save q_grasp from ik.py first."
        )

    out_root = Path(args.out)
    log_root = Path(args.logdir)

    if args.sweep:
        run_id = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
        out_root = out_root / "experiments" / run_id
        log_root = log_root / "experiments" / run_id
        results = []
        for trial in SWEEP_TRIALS:
            parameters = {**BASE_HYPERPARAMETERS, **trial["overrides"]}
            results.append(
                train_trial(
                    xml_path=xml_path,
                    seed=args.seed,
                    timesteps=args.sweep_timesteps,
                    n_envs=1,
                    out_dir=out_root / trial["name"],
                    log_dir=log_root / trial["name"],
                    hyperparameters=parameters,
                    trial_name=trial["name"],
                    sweep_mode=True,
                )
            )

        results.sort(
            key=lambda result: result["best_eval_mean_reward"], reverse=True
        )
        (out_root / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"Sweep results (ranked by best evaluation mean reward): {out_root}")
        for result in results:
            print(
                f"  {result['trial']}: "
                f"{result['best_eval_mean_reward']:.3f}"
            )
        return

    train_trial(
        xml_path=xml_path,
        seed=args.seed,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        out_dir=out_root,
        log_dir=log_root,
        hyperparameters=BASE_HYPERPARAMETERS,
        trial_name="PPO_baseline",
        sweep_mode=False,
    )


if __name__ == "__main__":
    main()
