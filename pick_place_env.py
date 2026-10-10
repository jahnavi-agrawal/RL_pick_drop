import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class PandaPickPlaceEnv(gym.Env):
    """
    PPO environment for the existing Panda MuJoCo scene.
    Expected model path by default:
      mujoco_menagerie/franka_emika_panda/panda_pick_place.xml

    Expected XML names:
      body "cube", site "drop_spot", hand body "hand"
    Panda arm joints: joint1 ... joint7
    """

    metadata = {"render_modes": []}

    def __init__(self, model_path=None, max_episode_steps=500, randomize_cube=False):
        super().__init__()
        here = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(
                here, "mujoco_menagerie", "franka_emika_panda", "panda_pick_place.xml"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MuJoCo XML not found: {model_path}\n"
                "Pass model_path=... or put this script in your proj1 folder."
            )

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.randomize_cube = randomize_cube
        self.steps = 0

        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_joint_ids = np.array(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names],
            dtype=int,
        )
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_ids]

        # Find the actuators that drive each arm joint by checking transmissions.
        self.arm_actuator_ids = []
        for jid in self.arm_joint_ids:
            matches = [
                aid for aid in range(self.model.nu)
                if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[aid, 0] == jid
            ]
            if not matches:
                raise RuntimeError(f"No actuator found for joint id {jid}. Check panda.xml.")
            self.arm_actuator_ids.append(matches[0])
        self.arm_actuator_ids = np.asarray(self.arm_actuator_ids, dtype=int)

        self.cube_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.drop_site_id = self._name_id(mujoco.mjtObj.mjOBJ_SITE, "drop_spot")
        self.hand_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Prefer a named end-effector site if the model has one.
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.has_ee_site = self.ee_site_id >= 0

        # The existing IK script used this offset if attachment_site is absent.
        self.local_grip = np.array([0.0, 0.0, 0.1094], dtype=float)

        # A free joint on the cube is expected. Find its qpos/dof addresses.
        first_joint = self.model.body_jntadr[self.cube_body_id]
        if first_joint < 0:
            raise RuntimeError("The 'cube' body must have a free joint.")
        self.cube_joint_id = int(first_joint)
        if self.model.jnt_type[self.cube_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("The first joint on body 'cube' must be a free joint.")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_dof_adr = int(self.model.jnt_dofadr[self.cube_joint_id])

        # Most versions of the existing project use one gripper actuator at index 7.
        # We try to find it by name, then fall back to the eighth actuator.
        gripper_candidates = []
        for aid in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            if "grip" in name.lower():
                gripper_candidates.append(aid)
        self.gripper_actuator_id = (
            gripper_candidates[-1] if gripper_candidates else (7 if self.model.nu > 7 else None)
        )
        if self.gripper_actuator_id is None:
            raise RuntimeError("Could not identify a gripper actuator.")

        self.grip_ctrl_range = self.model.actuator_ctrlrange[self.gripper_actuator_id].copy()
        self.grip_ctrl_limited = bool(self.model.actuator_ctrllimited[self.gripper_actuator_id])
        # Based on the latest ik.py diagnostic in this project: 0=closed, 255=open.
        # Override these two values below if your XML reports the opposite convention.
        self.gripper_close_ctrl = float(self.grip_ctrl_range[0]) if self.grip_ctrl_limited else 0.0
        self.gripper_open_ctrl = float(self.grip_ctrl_range[1]) if self.grip_ctrl_limited else 255.0

        # Panda home pose. The policy controls small increments around current joint targets.
        self.home_q = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.8], dtype=float)
        self.action_scale = 0.04  # radians per policy step
        self.control_substeps = 5
        self._joint_targets = self.home_q.copy()
        self._initial_cube_pos = np.array([0.55, -0.15, 0.18], dtype=float)
        self._cube_start_z = float(self._initial_cube_pos[2])

        # Observation: q(7), qvel(7), ee(3), cube(3), target(3), gripper_open(1) = 24.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        # Seven joint increments and one open/close command.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def _name_id(self, obj_type, name):
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(
                f"Required MuJoCo name '{name}' was not found in {self.model_path}. "
                "Check the names in panda_pick_place.xml."
            )
        return idx

    def _ee_pos(self):
        if self.has_ee_site:
            return self.data.site_xpos[self.ee_site_id].copy()
        hand_pos = self.data.xpos[self.hand_body_id].copy()
        hand_rot = self.data.xmat[self.hand_body_id].reshape(3, 3)
        return hand_pos + hand_rot @ self.local_grip

    def _cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def _target_pos(self):
        return self.data.site_xpos[self.drop_site_id].copy()

    def _observation(self):
        q = self.data.qpos[self.arm_qpos_adr]
        qvel = self.data.qvel[self.arm_dof_adr]
        ee = self._ee_pos()
        cube = self._cube_pos()
        target = self._target_pos()
        # This is a command-state indicator, not a sensor measurement.
        gripper_open = 1.0 if self._last_gripper_command == "open" else 0.0
        return np.concatenate([q, qvel, ee, cube, target, [gripper_open]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.arm_qpos_adr] = self.home_q
        self._joint_targets = self.home_q.copy()

        cube_pos = self._initial_cube_pos.copy()
        if self.randomize_cube:
            cube_pos[0] += self.np_random.uniform(-0.04, 0.04)
            cube_pos[1] += self.np_random.uniform(-0.04, 0.04)
        self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = cube_pos
        self.data.qpos[self.cube_qpos_adr + 3:self.cube_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 6] = 0.0

        self._last_gripper_command = "open"
        for aid, q in zip(self.arm_actuator_ids, self._joint_targets):
            self.data.ctrl[aid] = q
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_ctrl

        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.steps = 0
        self._cube_start_z = float(self._cube_pos()[2])
        self._max_lift_height = 0.0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float).clip(-1.0, 1.0)
        self.steps += 1

        # PPO proposes small joint target increments.
        self._joint_targets += self.action_scale * action[:7]
        for i, aid in enumerate(self.arm_actuator_ids):
            ctrl = self._joint_targets[i]
            if self.model.actuator_ctrllimited[aid]:
                lo, hi = self.model.actuator_ctrlrange[aid]
                ctrl = np.clip(ctrl, lo, hi)
            self.data.ctrl[aid] = ctrl

        # Last action dimension: positive opens, negative closes.
        if action[7] >= 0.0:
            self._last_gripper_command = "open"
            self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_ctrl
        else:
            self._last_gripper_command = "closed"
            self.data.ctrl[self.gripper_actuator_id] = self.gripper_close_ctrl

        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._observation()
        ee = self._ee_pos()
        cube = self._cube_pos()
        target = self._target_pos()

        ee_cube_dist = float(np.linalg.norm(ee - cube))
        cube_target_dist = float(np.linalg.norm(cube[:2] - target[:2]))
        lift_height = float(cube[2] - self._cube_start_z)
        self._max_lift_height = max(self._max_lift_height, lift_height)

        # Dense shaping: approach, lift, then move the lifted cube toward target.
        reward = -2.0 * ee_cube_dist
        reward += 3.0 * max(0.0, lift_height)
        if lift_height > 0.035:
            reward += 2.0 * (1.0 - np.tanh(4.0 * cube_target_dist))

        # Successful placement: close to target in XY, near tabletop height, gripper open.
        target_z = target[2]
        at_target_xy = cube_target_dist < 0.055
        near_table_height = abs(cube[2] - self._cube_start_z) < 0.045
        released = self._last_gripper_command == "open"
        success = bool(
            at_target_xy and near_table_height and released
            and self._max_lift_height > 0.035
        )

        if success:
            reward += 25.0

        # Penalize dropping the cube off the table / below the scene.
        fallen = bool(cube[2] < 0.04)
        if fallen:
            reward -= 10.0

        terminated = success or fallen
        truncated = self.steps >= self.max_episode_steps
        info = {
            "success": success,
            "ee_cube_dist": ee_cube_dist,
            "cube_target_dist": cube_target_dist,
            "lift_height": lift_height,
            "max_lift_height": self._max_lift_height,
            "cube_pos": cube.copy(),
            "target_pos": target.copy(),
        }
        return obs, float(reward), terminated, truncated, info
