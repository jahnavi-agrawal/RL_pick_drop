"""Watch a trained lift-focused PPO policy in the MuJoCo viewer.

On macOS, launch this with mjpython:
    mjpython watch_ppo_lift_focus.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from train_ppo_lift_focus import LiftFocusedPandaEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/panda_lift_focus_v1/best/best_model.zip",
        help="PPO checkpoint to load (best_model.zip or final_model.zip).",
    )
    parser.add_argument(
        "--xml",
        default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--delay", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train with train_ppo_lift_focus.py first, or pass --model with "
            "the path to a saved checkpoint."
        )

    model = PPO.load(str(model_path), device="cpu")
    env = LiftFocusedPandaEnv(xml_path=args.xml, render_mode="human")
    try:
        obs, _ = env.reset(seed=args.seed)
        print(f"Watching {model_path}")
        env.render()
        print("Viewer ready. Press F5 for fullscreen; policy starts in 5 seconds.")
        for seconds_left in range(5, 0, -1):
            print(f"Starting in {seconds_left}...")
            env.render()
            time.sleep(1.0)

        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if step % 10 == 0 or terminated or truncated:
                print(
                    f"step={step:3d} reward={reward:7.3f} "
                    f"lift={info['lift_height_m']:.3f} m "
                    f"bilateral={info['bilateral_contact']} "
                    f"hold={info['hold_steps']} "
                    f"success={info['success']} "
                    f"dropped={info['dropped']}"
                )

            if terminated or truncated:
                print("Episode ended. Close the viewer window to exit.")
                while env._viewer is not None and env._viewer.is_running():
                    env.render()
                    time.sleep(0.05)
                break

            if args.delay > 0:
                time.sleep(args.delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
