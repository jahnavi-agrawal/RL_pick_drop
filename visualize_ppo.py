import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from pick_place_env import PandaPickPlaceEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/panda_ppo_final.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of a trained PPO model.")
    parser.add_argument("--delay", type=float, default=0.015,
                        help="Pause between policy steps; increase to slow it down.")
    args = parser.parse_args()

    env = PandaPickPlaceEnv(randomize_cube=False)
    model = None if args.random else PPO.load(args.model)

    successes = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            print(f"\nEpisode {ep + 1}")
            while viewer.is_running() and not done:
                if args.random:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                viewer.sync()
                print(
                    f"step={env.steps:03d} reward={reward:+.3f} "
                    f"ee-cube={info['ee_cube_dist']:.3f}m "
                    f"lift={info['lift_height']:.3f}m "
                    f"cube-target={info['cube_target_dist']:.3f}m "
                    f"success={info['success']}",
                    flush=True,
                )
                time.sleep(args.delay)

            successes += int(info.get("success", False))
            print(f"Episode return={total_reward:.2f}; success={info.get('success', False)}")
            if not viewer.is_running():
                break

    print(f"\nSuccesses: {successes}/{args.episodes}")
    env.close()

if __name__ == "__main__":
    main()
