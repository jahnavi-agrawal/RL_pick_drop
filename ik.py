
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "panda_proj/panda_pick_place.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CUBE_POS = np.array([0.55, -0.15, 0.18])
PREGRASP_HEIGHT = 0.08

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Midpoint between the fingertips relative to the hand.
LOCAL_GRIP = np.array([0.0, 0.0, 0.1094])

# --------------------------------------------------
# FIND ROBOT JOINTS AND ACTUATORS
# --------------------------------------------------

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)

if hand_id < 0:
    raise ValueError("Could not find the Panda hand body.")

joint_ids = []
qpos_ids = []
dof_ids = []
arm_actuator_ids = []

for name in ARM_JOINTS:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Missing Panda joint: {name}")

    joint_ids.append(jid)
    qpos_ids.append(model.jnt_qposadr[jid])
    dof_ids.append(model.jnt_dofadr[jid])

    actuator_id = -1
    for aid in range(model.nu):
        if model.actuator_trnid[aid, 0] == jid:
            actuator_id = aid
            break

    if actuator_id < 0:
        raise ValueError(f"No actuator found for {name}")

    arm_actuator_ids.append(actuator_id)

# Find gripper actuators from the finger joint names.
finger_names = {"finger_joint1", "finger_joint2"}
gripper_actuator_ids = []

for aid in range(model.nu):
    jid = model.actuator_trnid[aid, 0]
    if jid < 0:
        continue

    joint_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, jid
    )

    if joint_name in finger_names:
        gripper_actuator_ids.append(aid)

if not gripper_actuator_ids:
    # Fallback for Menagerie models with a single named
    # gripper actuator.
    for name in ("actuator8", "gripper"):
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if aid >= 0:
            gripper_actuator_ids.append(aid)
            break

if not gripper_actuator_ids:
    raise ValueError(
        "No gripper actuator found. Inspect the actuator "
        "section in panda.xml."
    )

print("Arm actuator IDs:", arm_actuator_ids)
print("Gripper actuator IDs:", gripper_actuator_ids)

# Control endpoints for each gripper actuator.
# This assumes the low endpoint closes and the high
# endpoint opens the standard Panda position gripper.
gripper_ranges = {}

for aid in gripper_actuator_ids:
    if not model.actuator_ctrllimited[aid]:
        raise ValueError(
            f"Gripper actuator {aid} has no control limits. "
            "Set open/close targets for its actual actuator."
        )

    low, high = model.actuator_ctrlrange[aid]
    gripper_ranges[aid] = (float(low), float(high))
    print(
        f"Gripper actuator {aid}: "
        f"close={low}, open={high}"
    )


def set_gripper(open_gripper):
    for aid, (low, high) in gripper_ranges.items():
        data.ctrl[aid] = high if open_gripper else low


# --------------------------------------------------
# GRIPPER POSITION
# --------------------------------------------------

def get_gripper_position():
    hand_pos = data.xpos[hand_id]
    hand_rot = data.xmat[hand_id].reshape(3, 3)
    return hand_pos + hand_rot @ LOCAL_GRIP


# --------------------------------------------------
# INVERSE KINEMATICS
# --------------------------------------------------

def solve_ik(target, max_iterations=3000):
    q = data.qpos[qpos_ids].copy()

    for iteration in range(max_iterations):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)

        grip_pos = get_gripper_position()
        error = target - grip_pos

        if np.linalg.norm(error) < 0.003:
            print(
                f"IK converged in {iteration} iterations. "
                f"Error: {np.linalg.norm(error):.4f} m"
            )
            return q

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jac(
            model, data, jacp, jacr, grip_pos, hand_id
        )

        J = jacp[:, dof_ids]
        damping = 0.05

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            error
        )

        dq = np.clip(0.2 * dq, -0.03, 0.03)
        q += dq

        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)

    print(
        "Warning: IK did not fully converge. "
        f"Remaining error: {np.linalg.norm(target - get_gripper_position()):.4f} m"
    )
    return q


# --------------------------------------------------
# MOVE ARM TO A JOINT TARGET
# --------------------------------------------------

def move_to(q_target, duration, viewer):
    start = time.time()

    while time.time() - start < duration and viewer.is_running():
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_target[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)


# --------------------------------------------------
# RUN PICK APPROACH AND GRIP
# --------------------------------------------------

mujoco.mj_forward(model, data)

# Compute approach and grasp targets.
pregrasp_pos = CUBE_POS + np.array(
    [0.0, 0.0, PREGRASP_HEIGHT]
)

print("Solving pre-grasp IK...")
q_pregrasp = solve_ik(pregrasp_pos)

print("Solving cube-centre IK...")
q_grasp = solve_ik(CUBE_POS)

# Restore initial simulation state before the run.
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Opening gripper...")
    set_gripper(True)

    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()

    print("Moving above cube...")
    move_to(q_pregrasp, 4.0, viewer)

    print("Descending to cube...")
    move_to(q_grasp, 2.0, viewer)

    # Allow the arm to settle before closing.
    for _ in range(100):
        if not viewer.is_running():
            break
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_grasp[i]
        set_gripper(True)
        mujoco.mj_step(model, data)
        viewer.sync()

    print("Closing gripper...")
    set_gripper(False)

    for _ in range(300):
        if not viewer.is_running():
            break
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_grasp[i]
        set_gripper(False)
        mujoco.mj_step(model, data)
        viewer.sync()

    cube_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube"
    )

    print("Gripper command sent.")
    print("Cube position:", data.xpos[cube_id].copy())
    print("Gripper position:", get_gripper_position())

    print(
        "Inspect the fingers and cube in the viewer. "
        "The script will keep the viewer open."
    )

    while viewer.is_running():
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_grasp[i]
        set_gripper(False)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)