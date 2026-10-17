"""
Force-aware Panda lift environment (CPU-compatible).

Expected scene:
- A MuJoCo XML containing the Panda, a body named "cube" with a free joint,
  a body named "hand" (or configured end-effector body), and a site named
  "drop_spot" (used as the target position in the observation).
- The Panda arm has seven joints and seven arm actuators.
- The gripper has one actuator whose control range is typically 0..255
  (0=closed, 255=open in the standard Menagerie Panda model).

Important:
Finger-force extraction depends on the exact geom names in your XML.
Set LEFT_FINGER_KEY and RIGHT_FINGER_KEY below to substrings present in your
left/right finger geom names. Force is estimated from MuJoCo contact forces
between each finger and cube. If names do not match, the corresponding force
will remain zero.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# ---------- Configuration you may need to adjust to match your XML ----------
DEFAULT_XML = Path("mujoco_menagerie/franka_emika_panda/panda_pick_place.xml")
CUBE_BODY_NAME = "cube"
EE_BODY_NAME = "hand"
TARGET_SITE_NAME = "drop_spot"
GRIPPER_ACTUATOR_NAME = "actuator8"  # if absent, last actuator is used as fallback

# These are name fragments, not guaranteed names. Check your XML geom names.
LEFT_FINGER_KEY = "left_finger"
RIGHT_FINGER_KEY = "right_finger"

# Panda arm control and task settings
ARM_ACTION_SCALE_RAD = 0.035
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSE_CTRL = 0.0
SIM_STEPS_PER_ACTION = 5
LIFT_SUCCESS_HEIGHT = 0.05       # 5 cm above cube's initial height
SUCCESS_HOLD_STEPS = 10           # consecutive environment actions
FALL_Z_THRESHOLD = 0.05           # world z below this is treated as a fall

# Reward weights: initial experimental hypotheses, not optimized values
W_APPROACH_PROGRESS = 5.0
W_DISTANCE = 0.05
R_FIRST_FINGER_CONTACT = 1.0
R_BILATERAL_CONTACT = 2.0
W_LIFT_PROGRESS = 20.0
R_HOLD_PER_STEP = 0.5
R_SUCCESS = 25.0
R_DROP = -10.0

# Force reward parameters (measure forces in your scene before tuning these)
TARGET_GRIP_FORCE_N = 2.0
W_FORCE_TARGET = 0.05
MAX_GRIP_FORCE_N = 8.0
W_EXCESS_FORCE = 0.1


def _name_or_none(model, kind: str, name: str) -> Optional[int]:
    """Resolve a MuJoCo object name to its integer id, or return None."""
    obj_enum = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }[kind]
    return mujoco.mj_name2id(model, obj_enum, name)


class PandaLiftEnv(gym.Env):
    """
    Observation (26D):
      q(7), qvel(7), end-effector xyz(3), cube xyz(3), target xyz(3),
      gripper opening/control normalized(1), left finger force(1), right force(1).
    Action (8D, each clipped to [-1, 1]):
      seven incremental arm joint-target commands + one gripper open/close command.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, xml_path=DEFAULT_XML, render_mode=None):
        super().__init__()
        self.xml_path = Path(xml_path)
        self.render_mode = render_mode
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML not found: {self.xml_path.resolve()}\n"
                "Pass --xml with the path to your panda_pick_place.xml."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path.resolve()))
        self.data = mujoco.MjData(self.model)

        self.cube_body_id = _name_or_none(self.model, "body", CUBE_BODY_NAME)
        self.ee_body_id = _name_or_none(self.model, "body", EE_BODY_NAME)
        self.target_site_id = _name_or_none(self.model, "site", TARGET_SITE_NAME)
        if self.cube_body_id is None:
            raise ValueError(f"Body '{CUBE_BODY_NAME}' not found in XML.")
        if self.ee_body_id is None:
            raise ValueError(
                f"End-effector body '{EE_BODY_NAME}' not found. "
                "Set EE_BODY_NAME to the actual hand/end-effector body name in your XML."
            )
        if self.target_site_id is None:
            raise ValueError(
                f"Site '{TARGET_SITE_NAME}' not found. For lift-only, this position is not "
                "used by the reward, but it is retained in the 26D observation for compatibility."
            )

        # Identify the seven arm joints by the conventional Panda names.
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_joint_ids = []
        self.arm_qpos_addresses = []
        self.arm_dof_addresses = []
        for name in self.arm_joint_names:
            jid = _name_or_none(self.model, "joint", name)
            if jid is None:
                raise ValueError(f"Arm joint '{name}' not found in XML.")
            self.arm_joint_ids.append(jid)
            self.arm_qpos_addresses.append(int(self.model.jnt_qposadr[jid]))
            self.arm_dof_addresses.append(int(self.model.jnt_dofadr[jid]))

        # Match arm actuators to the joints they drive, using actuator_trnid.
        self.arm_actuator_ids = []
        for jid in self.arm_joint_ids:
            found = None
            for aid in range(self.model.nu):
                if int(self.model.actuator_trnid[aid, 0]) == jid:
                    found = aid
                    break
            if found is None:
                raise ValueError(
                    f"No actuator found for joint id {jid}. Check the actuator section in the XML."
                )
            self.arm_actuator_ids.append(found)

        gripper_id = _name_or_none(self.model, "actuator", GRIPPER_ACTUATOR_NAME)
        if gripper_id is None:
            if self.model.nu < 8:
                raise ValueError("Could not find gripper actuator and fewer than 8 actuators exist.")
            gripper_id = self.model.nu - 1
            print(
                f"WARNING: actuator '{GRIPPER_ACTUATOR_NAME}' not found; using final actuator "
                f"(id {gripper_id}). Verify this is the gripper actuator."
            )
        self.gripper_actuator_id = gripper_id

        self.left_finger_geom_ids = self._find_geom_ids(LEFT_FINGER_KEY)
        self.right_finger_geom_ids = self._find_geom_ids(RIGHT_FINGER_KEY)
        # if not self.left_finger_geom_ids:
        #     print(f"WARNING: no geom name contains '{LEFT_FINGER_KEY}'. Left force will be zero.")
        # if not self.right_finger_geom_ids:
        #     print(f"WARNING: no geom name contains '{RIGHT_FINGER_KEY}'. Right force will be zero.")

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        self.initial_cube_z = None
        self.previous_distance = None
        self.previous_lift = 0.0
        self.previous_contact_count = 0
        self.hold_counter = 0
        self.ever_lifted = False
        self.success_paid = False
        self.gripper_command = GRIPPER_OPEN_CTRL
        self.arm_targets = None
        self.elapsed_steps = 0
        self.max_episode_steps = 300

    def _find_geom_ids(self, name_fragment):
        ids = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name_fragment.lower() in name.lower():
                ids.append(gid)
        return set(ids)

    def _cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def _ee_pos(self):
        return self.data.xpos[self.ee_body_id].copy()

    def _target_pos(self):
        return self.data.site_xpos[self.target_site_id].copy()

    
    
    def _finger_forces(self):
        left_force = 0.0
        right_force = 0.0

        def body_id(name):
            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )

        cube_id = body_id("cube")
        left_id = body_id("left_finger")
        right_id = body_id("right_finger")

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]

            if b1 == cube_id:
                other = b2
            elif b2 == cube_id:
                other = b1
            else:
                continue

            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model, self.data, i, force
            )
            magnitude = float(np.linalg.norm(force[:3]))

            if other == left_id:
                left_force += magnitude
            elif other == right_id:
                right_force += magnitude

        return left_force, right_force

 
    def _gripper_opening_feature(self):
        lo = float(self.model.actuator_ctrlrange[self.gripper_actuator_id, 0])
        hi = float(self.model.actuator_ctrlrange[self.gripper_actuator_id, 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float(self.gripper_command / max(GRIPPER_OPEN_CTRL, 1.0))
        return float(np.clip((self.gripper_command - lo) / (hi - lo), 0.0, 1.0))

    def _observation(self):
        q = np.array([self.data.qpos[a] for a in self.arm_qpos_addresses], dtype=np.float32)
        qvel = np.array([self.data.qvel[a] for a in self.arm_dof_addresses], dtype=np.float32)
        ee = self._ee_pos()
        cube = self._cube_pos()
        target = self._target_pos()
        left_f, right_f = self._finger_forces()
        obs = np.concatenate([
            q, qvel, ee, cube, target,
            np.array([self._gripper_opening_feature(), left_f, right_f], dtype=np.float32)
        ]).astype(np.float32)
        return obs

    def _apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        current_q = np.array([self.data.qpos[a] for a in self.arm_qpos_addresses])
        if self.arm_targets is None:
            self.arm_targets = current_q.copy()
        self.arm_targets = self.arm_targets + ARM_ACTION_SCALE_RAD * action[:7]

        # Respect each joint's limited range when the joint has limits.
        for i, jid in enumerate(self.arm_joint_ids):
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                self.arm_targets[i] = np.clip(self.arm_targets[i], lo, hi)
            aid = self.arm_actuator_ids[i]
            ctrl_lo, ctrl_hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(self.arm_targets[i], ctrl_lo, ctrl_hi)

        # Gripper command is discrete-ish but controlled by the 8th continuous action.
        if action[7] > 0.2:
            self.gripper_command = GRIPPER_OPEN_CTRL
        elif action[7] < -0.2:
            self.gripper_command = GRIPPER_CLOSE_CTRL
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_command

    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Apply the Panda's home keyframe.
        home_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )

        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, home_id
            )

        # Restore the cube's intended initial pose.
        cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_free",
        )

        if cube_joint_id < 0:
            raise ValueError("Joint 'cube_free' not found.")

        cube_qpos_adr = self.model.jnt_qposadr[cube_joint_id]

        self.data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = [
            0.55, -0.15, 0.18, 1.0, 0.0, 0.0, 0.0
        ]

        # Forward kinematics and collision update.
        mujoco.mj_forward(self.model, self.data)

        # Initialize episode state.
        self.initial_cube_z = float(self._cube_pos()[2])
        self.arm_targets = np.array([
            self.data.qpos[a] for a in self.arm_qpos_addresses
        ])

        self.gripper_command = GRIPPER_OPEN_CTRL
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_command

        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = self.arm_targets[i]

        mujoco.mj_forward(self.model, self.data)

        ee = self._ee_pos()
        cube = self._cube_pos()

        self.previous_distance = float(np.linalg.norm(ee - cube))
        self.previous_lift = 0.0
        self.previous_contact_count = 0
        self.hold_counter = 0
        self.ever_lifted = False
        self.success_paid = False
        self.elapsed_steps = 0

        return self._observation(), {}


    def step(self, action):
        ee_before = self._ee_pos()
        cube_before = self._cube_pos()
        distance_before = float(np.linalg.norm(ee_before - cube_before))
        lift_before = max(0.0, float(cube_before[2] - self.initial_cube_z))

        self._apply_action(action)
        for _ in range(SIM_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self.elapsed_steps += 1
        ee_after = self._ee_pos()
        cube_after = self._cube_pos()
        distance_after = float(np.linalg.norm(ee_after - cube_after))
        lift_after = max(0.0, float(cube_after[2] - self.initial_cube_z))
        left_f, right_f = self._finger_forces()

        # Contact evidence: each finger must register at least a small force.
        contact_threshold = 0.1
        left_contact = left_f > contact_threshold
        right_contact = right_f > contact_threshold
        bilateral_contact = left_contact and right_contact
        contact_count = int(left_contact) + int(right_contact)

        # 1) Approach progress and small remaining-distance penalty.
        reward = W_APPROACH_PROGRESS * (distance_before - distance_after)
        reward -= W_DISTANCE * distance_after

        # 2) One-time contact bonuses (avoid paying the same bonus every step).
        if contact_count >= 1 and self.previous_contact_count == 0:
            reward += R_FIRST_FINGER_CONTACT
        if bilateral_contact and self.previous_contact_count < 2:
            reward += R_BILATERAL_CONTACT

        # 3) Reward upward progress only when both finger contacts are present.
        lift_progress = max(0.0, lift_after - lift_before)
        if bilateral_contact:
            reward += W_LIFT_PROGRESS * lift_progress

        # 4) Force regularisation only while both fingers contact the cube.
        if bilateral_contact:
            grip_force = 0.5 * (left_f + right_f)
            reward -= W_FORCE_TARGET * (grip_force - TARGET_GRIP_FORCE_N) ** 2
            excess = max(0.0, grip_force - MAX_GRIP_FORCE_N)
            reward -= W_EXCESS_FORCE * excess ** 2

        # 5) Hold bonus: cube is >= 5 cm up and bilateral contact persists.
        if lift_after >= LIFT_SUCCESS_HEIGHT and bilateral_contact:
            self.hold_counter += 1
            self.ever_lifted = True
            reward += R_HOLD_PER_STEP
        else:
            self.hold_counter = 0

        terminated = False
        success = False

        # One-time terminal success bonus after sustained hold.
        if self.hold_counter >= SUCCESS_HOLD_STEPS and not self.success_paid:
            reward += R_SUCCESS
            self.success_paid = True
            success = True
            terminated = True

        # Drop penalty: after a lift, falling back below the threshold ends the episode.
        if self.ever_lifted and lift_after < 0.02 and not success:
            reward += R_DROP
            terminated = True
        elif cube_after[2] < FALL_Z_THRESHOLD and not success:
            # Use elif so one fall cannot accidentally receive the drop penalty twice.
            reward += R_DROP
            terminated = True

        truncated = self.elapsed_steps >= self.max_episode_steps
        self.previous_distance = distance_after
        self.previous_lift = lift_after
        self.previous_contact_count = contact_count

        info = {
            "success": success,
            "lift_height_m": float(lift_after),
            "gripper_cube_distance_m": float(distance_after),
            "left_finger_force_N": float(left_f),
            "right_finger_force_N": float(right_f),
            "bilateral_contact": bool(bilateral_contact),
            "hold_steps": int(self.hold_counter),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        # Use the passive viewer for interactive visualisation.
        if not hasattr(self, "_viewer") or self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        viewer = getattr(self, "_viewer", None)
        if viewer is not None:
            viewer.close()
            self._viewer = None
