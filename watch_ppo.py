
import time
from pathlib import Path

from stable_baselines3 import PPO
from pick_lift_env import PandaLiftEnv

XML_PATH = "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml"

# Change this if your model was saved somewhere else.
MODEL_PATH = "runs/panda_lift_baseline/best/best_model.zip"

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Update MODEL_PATH to your saved model."
        )

    model = PPO.load(MODEL_PATH, device="cpu")

    # Uses the same IK-derived pregrasp start pose as training.
    env = PandaLiftEnv(xml_path=XML_PATH, render_mode="human")

    try:
        obs, _ = env.reset(seed=123)

        import numpy as np
        import mujoco

        data = env.unwrapped.data
        mj_model = env.unwrapped.model

        cube_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        hand_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        left_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        right_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )

        print("Cube:", data.xpos[cube_id])
        print("Hand:", data.xpos[hand_id])
        print("Left finger:", data.xpos[left_id])
        print("Right finger:", data.xpos[right_id])
        print("Left-to-cube distance:",
            np.linalg.norm(data.xpos[left_id] - data.xpos[cube_id]))
        print("Right-to-cube distance:",
            np.linalg.norm(data.xpos[right_id] - data.xpos[cube_id]))

        print("Viewer opened. Watching trained PPO policy...")

        for step in range(300):
            # Deterministic actions make evaluation repeatable.
            action, _ = model.predict(obs, deterministic=True)

            print(
                f"Step {step}: gripper_action={action[7]:.3f}"
                if step % 10 == 0 else "",
                end="\n" if step % 10 == 0 else "",
                )

            obs, reward, terminated, truncated, info = env.step(action)
            if step % 20 == 0:
                env.debug_cube_contacts()

            # Update the MuJoCo viewer.
            env.render()

            if step % 10 == 0:
                print(
                    f"Step={step:3d} | "
                    f"lift={info['lift_height_m']:.3f} m | "
                    f"distance={info['gripper_cube_distance_m']:.3f} m | "
                    f"left_force={info['left_finger_force_N']:.2f} N | "
                    f"right_force={info['right_finger_force_N']:.2f} N | "
                    f"success={info['success']}"
                )

            time.sleep(0.02)  # Slow playback so you can watch it.

            if terminated or truncated:
                print(
                    f"Episode ended: success={info['success']}, "
                    f"final lift={info['lift_height_m']:.3f} m"
                )
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
