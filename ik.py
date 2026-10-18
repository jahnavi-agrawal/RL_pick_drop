
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml"

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CUBE_POS = np.array([0.55, -0.15, 0.18])
PREGRASP_HEIGHT = 0.10
APPROACH_HEIGHT = 0.003

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Midpoint between the fingertips relative to the hand.
LOCAL_GRIP = np.array([0.0, 0.0, 0.1094])

# Initial orientation hypothesis:
# hand local +Z points downward in world coordinates.
# Local Y is aligned with world -Y.
TARGET_ROT = np.diag([1.0, -1.0, -1.0])

POSITION_TOL = 0.005       # metres
ORIENTATION_TOL = 0.08     # radians
DAMPING = 0.05
ORIENTATION_WEIGHT = 0.15

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# --------------------------------------------------
# FIND ROBOT JOINTS AND ACTUATORS
# --------------------------------------------------

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)
cube_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "cube"
)

if hand_id < 0:
    raise ValueError("Could not find body 'hand'.")
if cube_id < 0:
    raise ValueError("Could not find body 'cube'.")

joint_ids = []
qpos_ids = []
dof_ids = []
arm_actuator_ids = []

for name in ARM_JOINTS:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Missing joint: {name}")

    joint_ids.append(jid)
    qpos_ids.append(model.jnt_qposadr[jid])
    dof_ids.append(model.jnt_dofadr[jid])

    aid = -1
    for candidate in range(model.nu):
        if model.actuator_trnid[candidate, 0] == jid:
            aid = candidate
            break

    if aid < 0:
        raise ValueError(f"No actuator found for {name}")

    arm_actuator_ids.append(aid)

# Find individual finger actuators, or the combined gripper actuator.
finger_names = {"finger_joint1", "finger_joint2"}
gripper_actuator_ids = []

for aid in range(model.nu):
    jid = model.actuator_trnid[aid, 0]
    if jid < 0:
        continue

    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, jid
    )
    if name in finger_names:
        gripper_actuator_ids.append(aid)

if not gripper_actuator_ids:
    for name in ("actuator8", "gripper"):
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if aid >= 0:
            gripper_actuator_ids.append(aid)
            break

if not gripper_actuator_ids:
    raise ValueError("Could not find a gripper actuator.")

gripper_ranges = {}
for aid in gripper_actuator_ids:
    if not model.actuator_ctrllimited[aid]:
        raise ValueError(
            f"Gripper actuator {aid} has no control limits."
        )
    low, high = model.actuator_ctrlrange[aid]
    gripper_ranges[aid] = (float(low), float(high))

print("Arm actuator IDs:", arm_actuator_ids)
print("Gripper actuator IDs:", gripper_actuator_ids)
print("Gripper ranges:", gripper_ranges)


def set_gripper(open_gripper):
    for aid, (low, high) in gripper_ranges.items():
        data.ctrl[aid] = high if open_gripper else low


# --------------------------------------------------
# RESET TO HOME AND RESTORE CUBE
# --------------------------------------------------

def reset_scene():
    mujoco.mj_resetData(model, data)

    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)

    # A keyframe may also overwrite the free cube joint.
    cube_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
    )
    if cube_joint_id < 0:
        raise ValueError("Could not find joint 'cube_free'.")

    cube_adr = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_adr:cube_adr + 7] = [
        *CUBE_POS, 1.0, 0.0, 0.0, 0.0
    ]

    mujoco.mj_forward(model, data)
    set_gripper(True)

    for i, aid in enumerate(arm_actuator_ids):
        data.ctrl[aid] = data.qpos[qpos_ids[i]]

    mujoco.mj_forward(model, data)


# --------------------------------------------------
# GRIPPER POSITION
# --------------------------------------------------

def get_gripper_position():
    hand_pos = data.xpos[hand_id]
    hand_rot = data.xmat[hand_id].reshape(3, 3)
    return hand_pos + hand_rot @ LOCAL_GRIP


# --------------------------------------------------
# POSITION + ORIENTATION IK
# --------------------------------------------------

def solve_ik(target_pos, target_rot=TARGET_ROT, max_iterations=4000):
    q = data.qpos[qpos_ids].copy()
    target_rot = np.asarray(target_rot, dtype=float)

    for iteration in range(max_iterations):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)

        grip_pos = get_gripper_position()
        current_rot = data.xmat[hand_id].reshape(3, 3)

        pos_error = target_pos - grip_pos

        # Robust axis-angle orientation error, including near 180 degrees.
        rot_error = Rotation.from_matrix(
            target_rot @ current_rot.T
        ).as_rotvec()

        if (
            np.linalg.norm(pos_error) < POSITION_TOL
            and np.linalg.norm(rot_error) < ORIENTATION_TOL
        ):
            print(
                f"IK converged in {iteration} iterations | "
                f"position error={np.linalg.norm(pos_error):.4f} m | "
                f"orientation error={np.linalg.norm(rot_error):.4f} rad"
            )
            return q.copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jac(
            model, data, jacp, jacr, grip_pos, hand_id
        )

        Jp = jacp[:, dof_ids]
        Jr = jacr[:, dof_ids]

        J = np.vstack([
            Jp,
            ORIENTATION_WEIGHT * Jr
        ])
        error = np.concatenate([
            pos_error,
            ORIENTATION_WEIGHT * rot_error
        ])

        dq = J.T @ np.linalg.solve(
            J @ J.T + DAMPING**2 * np.eye(6),
            error
        )

        dq = np.clip(0.2 * dq, -0.03, 0.03)
        q += dq

        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q[i] = np.clip(q[i], low, high)

    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)

    pos_err = np.linalg.norm(target_pos - get_gripper_position())
    rot_err = np.linalg.norm(
        Rotation.from_matrix(
            target_rot @ data.xmat[hand_id].reshape(3, 3).T
        ).as_rotvec()
    )

    print(
        "WARNING: IK did not meet both tolerances | "
        f"position error={pos_err:.4f} m | "
        f"orientation error={rot_err:.4f} rad"
    )
    return q.copy()


def check_solution(q, target_pos, label):
    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)

    pos_err = np.linalg.norm(target_pos - get_gripper_position())
    rot_err = np.linalg.norm(
        Rotation.from_matrix(
            TARGET_ROT @ data.xmat[hand_id].reshape(3, 3).T
        ).as_rotvec()
    )

    print(
        f"{label}: position error={pos_err:.4f} m, "
        f"orientation error={rot_err:.4f} rad"
    )


# --------------------------------------------------
# MOVE ARM
# --------------------------------------------------

def move_to(q_target, duration, viewer):
    start = time.time()

    while time.time() - start < duration and viewer.is_running():
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_target[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)


def hold_pose(q_target, steps, viewer, open_gripper):
    for _ in range(steps):
        if not viewer.is_running():
            break

        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_target[i]

        set_gripper(open_gripper)
        mujoco.mj_step(model, data)
        viewer.sync()


# --------------------------------------------------
# SOLVE AND VALIDATE
# --------------------------------------------------

reset_scene()
home_q = data.qpos[qpos_ids].copy()

pregrasp_pos = CUBE_POS + np.array(
    [0.0, 0.0, PREGRASP_HEIGHT]
)

# The fingertip midpoint should reach the cube centre height
# to grasp its sides. Approach from above first.
grasp_pos = CUBE_POS + np.array(
    [0.0, 0.0, APPROACH_HEIGHT]
)

print("\nSolving pre-grasp IK...")
data.qpos[qpos_ids] = home_q
mujoco.mj_forward(model, data)
q_pregrasp = solve_ik(pregrasp_pos)
np.save("pregrasp_q.npy", q_pregrasp)
print("Saved pre-grasp joint configuration.")
check_solution(q_pregrasp, pregrasp_pos, "Pre-grasp")

print("\nSolving grasp IK...")
data.qpos[qpos_ids] = q_pregrasp
mujoco.mj_forward(model, data)
q_grasp = solve_ik(grasp_pos)
check_solution(q_grasp, grasp_pos, "Grasp")

# Restore the scene before executing the motion.
reset_scene()

# --------------------------------------------------
# EXECUTE PICK TEST
# --------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\nOpening gripper...")
    set_gripper(True)

    for _ in range(100):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()

    print("Moving above cube...")
    move_to(q_pregrasp, 4.0, viewer)
    hold_pose(q_pregrasp, 100, viewer, open_gripper=True)

    print("Descending toward cube...")
    move_to(q_grasp, 2.0, viewer)
    hold_pose(q_grasp, 100, viewer, open_gripper=True)

    print("Closing gripper...")
    hold_pose(q_grasp, 500, viewer, open_gripper=False)

    print("Lifting cube...")

    q_lift = q_grasp.copy()
    q_lift[1] -= 0.25  # Temporary test only; may not move vertically.

    move_to(q_lift, 2.0, viewer)
    hold_pose(q_lift, 200, viewer, open_gripper=False)

    print("Cube position after lift:", data.xpos[cube_id].copy())

    print("Cube position:", data.xpos[cube_id].copy())
    print("Hand position:", data.xpos[hand_id].copy())
    print("Gripper position:", get_gripper_position())
    print(
        "Inspect finger alignment and cube contact in the viewer. "
        "Close the viewer to exit."
    )

    while viewer.is_running():
        for i, aid in enumerate(arm_actuator_ids):
            data.ctrl[aid] = q_grasp[i]

        set_gripper(False)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
