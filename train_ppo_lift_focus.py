"""Train PPO with a reward focused on grasping, lifting, and holding.

Run from the project root:
    python train_ppo_lift_focus.py

This is an alternative to train_ppo_cpu.py. It starts from grasp_q.npy, so it
does not pay for moving the hand toward the cube. Models and logs use separate
directories and do not replace the baseline run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from pick_lift_env import (
    FALL_Z_THRESHOLD,
    LIFT_SUCCESS_HEIGHT,
    SIM_STEPS_PER_ACTION,
    SUCCESS_HOLD_STEPS,
    PandaLiftEnv,
)


# Conservative starting weights for this reward variant; treat them as tuning
# values, not validated physical constants.


class LiftFocusedPandaEnv(PandaLiftEnv):
    """PandaLiftEnv with no approach shaping and a capped lift-progress reward."""

    def step(self, action):
        cube_before = self._cube_pos()
        lift_before = max(0.0, float(cube_before[2] - self.initial_cube_z))

        self._apply_action(action)
        for _ in range(SIM_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self.elapsed_steps += 1
        cube_after = self._cube_pos()
        lift_after = max(0.0, float(cube_after[2] - self.initial_cube_z))
        distance_after = float(np.linalg.norm(self._ee_pos() - cube_after))
        left_force, right_force = self._finger_forces()

        left_contact = left_force > CONTACT_FORCE_THRESHOLD_N
        right_contact = right_force > CONTACT_FORCE_THRESHOLD_N
        bilateral_contact = left_contact and right_contact
        contact_count = int(left_contact) + int(right_contact)

        # Starting at the IK grasp pose means approach distance is not rewarded.
        reward = 0.0

        # Pay once when the second finger establishes contact.
        if bilateral_contact and self.previous_contact_count < 2:
            reward += CONTACT_BONUS

        # Reward positive lift progress only during a two-finger grasp. Capping
        # the per-step change limits reward spikes from contact impulses.
        if bilateral_contact:
            lift_progress = min(max(0.0, lift_after - lift_before),
                                MAX_LIFT_REWARD_METERS_PER_STEP)
            reward += LIFT_REWARD_PER_METER * lift_progress

            mean_force = 0.5 * (left_force + right_force)
            excess_force = max(0.0, mean_force - FORCE_LIMIT_N)
            reward -= EXCESS_FORCE_PENALTY * excess_force**2

        if lift_after >= LIFT_SUCCESS_HEIGHT and bilateral_contact:
            self.hold_counter += 1
            self.ever_lifted = True
            reward += HOLD_BONUS_PER_STEP
        else:
            self.hold_counter = 0

        terminated = False
        success = False
        if self.hold_counter >= SUCCESS_HOLD_STEPS and not self.success_paid:
            reward += SUCCESS_BONUS
            self.success_paid = True
            success = True
            terminated = True

        if self.ever_lifted and lift_after < 0.02 and not success:
            reward += DROP_PENALTY
            terminated = True
        elif cube_after[2] < FALL_Z_THRESHOLD and not success:
            reward += DROP_PENALTY
            terminated = True

        truncated = self.elapsed_steps >= self.max_episode_steps
        self.previous_distance = distance_after
        self.previous_lift = lift_after
        self.previous_contact_count = contact_count

        info = {
            "success": success,
            "is_success": success,
            "dropped": bool(terminated and not success),
            "lift_height_m": float(lift_after),
            "gripper_cube_distance_m": distance_after,
            "left_finger_force_N": float(left_force),
            "right_finger_force_N": float(right_force),
            "bilateral_contact": bool(bilateral_contact),
            "hold_steps": int(self.hold_counter),
        }
        return self._observation(), float(reward), terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--out", default="runs/panda_lift_focus_v1")
    parser.add_argument("--logdir", default="logs/panda_lift_focus_v1")
    return parser.parse_args()


def evaluate_policy(model, xml_path: str, seed: int, episodes: int):
    """Collect task metrics that EvalCallback's mean reward does not show."""
    eval_env = LiftFocusedPandaEnv(xml_path=xml_path)
    episode_results = []
    try:
        for episode in range(episodes):
            obs, _ = eval_env.reset(seed=seed + episode)
            episode_return = 0.0
            max_lift = 0.0
            bilateral_steps = 0
            dropped = False
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_return += float(reward)
                max_lift = max(max_lift, info["lift_height_m"])
                bilateral_steps += int(info["bilateral_contact"])
                dropped = dropped or info["dropped"]
                steps += 1
                if terminated or truncated:
                    break

            episode_results.append({
                "episode": episode,
                "return": episode_return,
                "success": bool(info["success"]),
                "max_lift_m": float(max_lift),
                "bilateral_contact_step_rate": bilateral_steps / max(steps, 1),
                "dropped": bool(dropped),
                "steps": steps,
            })
    finally:
        eval_env.close()

    return {
        "episodes": episodes,
        "success_count": sum(result["success"] for result in episode_results),
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "mean_return": float(np.mean([r["return"] for r in episode_results])),
        "mean_max_lift_m": float(np.mean([r["max_lift_m"] for r in episode_results])),
        "bilateral_contact_step_rate": float(
            np.mean([r["bilateral_contact_step_rate"] for r in episode_results])
        ),
        "drop_count": sum(result["dropped"] for result in episode_results),
        "per_episode": episode_results,
    }


def main():
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1.")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be at least 1.")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1.")
    torch.set_num_threads(args.torch_threads)

    xml_path = str(Path(args.xml).resolve())
    if not Path(xml_path).is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    if not Path("grasp_q.npy").is_file():
        raise FileNotFoundError(
            "grasp_q.npy was not found in the project root. Run ik.py first."
        )

    out_dir = Path(args.out)
    log_dir = Path(args.logdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        LiftFocusedPandaEnv,
        n_envs=1,
        seed=args.seed,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "train"),
    )
    eval_env = make_vec_env(
        LiftFocusedPandaEnv,
        n_envs=1,
        seed=args.seed + 1000,
        env_kwargs={"xml_path": xml_path},
        monitor_dir=str(log_dir / "eval"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "evaluations"),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_lift_focus",
    )

    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "xml_path": xml_path,
        "total_timesteps": args.timesteps,
        "n_envs": 1,
        "seed": args.seed,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "start_state": "grasp_q.npy",
        "reward": {
            "approach_distance": 0.0,
            "bilateral_contact_once": CONTACT_BONUS,
            "lift_per_meter": LIFT_REWARD_PER_METER,
            "lift_progress_cap_m_per_step": MAX_LIFT_REWARD_METERS_PER_STEP,
            "hold_per_step": HOLD_BONUS_PER_STEP,
            "success": SUCCESS_BONUS,
            "drop": DROP_PENALTY,
            "force_limit_N": FORCE_LIMIT_N,
            "excess_force_penalty": EXCESS_FORCE_PENALTY,
        },
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

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
            tb_log_name="PPO_lift_focus",
        )
        model.save(str(out_dir / "final_model"))
        print(f"Final model saved to: {out_dir / 'final_model.zip'}")
        print(f"Best model saved to: {out_dir / 'best' / 'best_model.zip'}")

        best_model_path = out_dir / "best" / "best_model.zip"
        eval_model = PPO.load(
            str(best_model_path if best_model_path.is_file()
                else out_dir / "final_model.zip"),
            device="cpu",
        )
        evaluation = evaluate_policy(
            eval_model,
            xml_path=xml_path,
            seed=args.seed + 2000,
            episodes=args.eval_episodes,
        )
        (out_dir / "evaluation_summary.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        print(
            f"Final evaluation: success={evaluation['success_rate']:.1%} "
            f"({evaluation['success_count']}/{evaluation['episodes']}), "
            f"mean max lift={evaluation['mean_max_lift_m']:.3f} m, "
            f"bilateral contact={evaluation['bilateral_contact_step_rate']:.1%}, "
            f"drops={evaluation['drop_count']}"
        )
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
