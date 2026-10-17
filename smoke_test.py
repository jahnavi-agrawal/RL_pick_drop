"""Quick smoke test: validates XML names, observation and action shapes."""
import argparse
from pick_lift_env import PandaLiftEnv

parser = argparse.ArgumentParser()
parser.add_argument("--xml", default="mujoco_menagerie/franka_emika_panda/panda_pick_place.xml")
args = parser.parse_args()

env = PandaLiftEnv(xml_path=args.xml)
obs, info = env.reset(seed=7)
print("Initial hand position:", env._ee_pos())
print("Initial cube position:", env._cube_pos())
print(
    "Initial arm joint positions:",
    [env.data.qpos[a] for a in env.arm_qpos_addresses]
)
print("Initial gripper control:", env.data.ctrl[env.gripper_actuator_id])

print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)
print("Initial cube position:", env._cube_pos())
print("Initial finger forces (N):", env._finger_forces())
obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 0, 0, -1])
print("One-step reward:", reward)
print("Next observation shape:", obs.shape)
print("Step diagnostics:", info)
env.close()
