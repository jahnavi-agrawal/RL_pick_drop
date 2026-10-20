"""Open a frozen MuJoCo view of the Panda and cube.

On macOS run this with MuJoCo's Python launcher:
    mjpython view_static_scene.py

Use --pose home for the XML home keyframe or --pose grasp to load grasp_q.npy.
The viewer is static: no policy runs and the simulation does not step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_XML = "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml"
CUBE_POSITION = np.array([0.55, -0.15, 0.18], dtype=np.float64)
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def object_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}.")
    return object_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--pose", choices=("grasp", "home"), default="grasp")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = Path(args.xml).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    cube_joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos = int(model.jnt_qposadr[cube_joint_id])
    data.qpos[cube_qpos:cube_qpos + 7] = [*CUBE_POSITION, 1.0, 0.0, 0.0, 0.0]

    arm_qpos_addresses = []
    arm_actuator_ids = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        arm_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        actuator_id = next(
            (
                aid for aid in range(model.nu)
                if int(model.actuator_trnid[aid, 0]) == joint_id
            ),
            None,
        )
        if actuator_id is None:
            raise ValueError(f"No actuator is connected to {joint_name!r}.")
        arm_actuator_ids.append(actuator_id)

    if args.pose == "grasp":
        grasp_path = Path("grasp_q.npy")
        if not grasp_path.is_file():
            raise FileNotFoundError(
                "grasp_q.npy not found in the project root. "
                "Run mjpython ik.py first or use --pose home."
            )
        grasp_q = np.asarray(np.load(grasp_path), dtype=np.float64)
        if grasp_q.shape != (7,):
            raise ValueError(f"Expected seven joint values in {grasp_path}; got {grasp_q.shape}.")
        data.qpos[arm_qpos_addresses] = grasp_q

    # Keep actuator controls consistent with the displayed joint positions.
    for qpos_address, actuator_id in zip(arm_qpos_addresses, arm_actuator_ids):
        data.ctrl[actuator_id] = data.qpos[qpos_address]
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
    )
    if gripper_id >= 0:
        data.ctrl[gripper_id] = 255.0

    mujoco.mj_forward(model, data)
    print(f"Scene loaded: {xml_path}")
    print(f"Arm pose: {args.pose}; cube position: {data.xpos[object_id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')]}")
    print("Close the MuJoCo viewer window to exit. The scene will remain frozen.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
