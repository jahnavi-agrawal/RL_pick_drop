# Detailed Walkthrough: MuJoCo Panda Lift Environment and PPO

This guide explains where the robot scene is defined, where state/action/reward are defined in code, and how the environment advances in MuJoCo. It is written to let you follow the experiment without having to reverse-engineer every function.

## 1. Big picture: which file does what?

| File | Purpose |
|---|---|
| `mujoco_menagerie/franka_emika_panda/panda_pick_place.xml` | The scene: robot, table, cube, sites/markers, actuators, initial positions and physics parameters. |
| `pick_lift_env.py` | The learning environment: state/observation, action mapping, contact forces, reward, success and drop conditions. |
| `train_ppo_cpu.py` | Creates the environment and trains PPO on CPU. It does not define the reward itself. |
| `models/panda_lift_ppo.zip` | Saved trained policy after training finishes. |

The XML describes the physical simulation. Python describes how an agent observes and controls that simulation and how the outcome is scored.

## 2. Where are positions defined in MuJoCo?

Open `mujoco_menagerie/franka_emika_panda/panda_pick_place.xml`.

The XML discussed in our project has a Panda include and scene objects such as the table, cube, pick marker and drop marker. Typical definitions look like:

```xml
<include file="panda.xml"/>

<body name="my_table" pos="0.7 0 0.05">
    ...
</body>

<body name="cube" pos="0.55 -0.15 0.18">
    <freejoint name="cube_free"/>
    ...
</body>

<site name="pick_spot" pos="0.55 -0.15 0.152" .../>
<site name="drop_spot" pos="0.55 0.20 0.152" .../>
```

Use the actual values in your current XML; the snippet above is illustrative of the scene values previously discussed.

### What does `pos` mean?

- For a body, `pos="x y z"` sets the body's position relative to its parent body.
- For a site inside a body, `pos` is relative to that site's parent body.
- A free-jointed cube can move during simulation. Its XML `pos` is its initial pose; after the simulation starts, the live position is read from MuJoCo's data.

### Where is the robot gripper position defined?

The robot and hand geometry are generally loaded through:

```xml
<include file="panda.xml"/>
```

Then open the included `panda.xml` and follow the nested bodies/joints toward the hand/fingers. The hand's position is determined by the chain of joint positions, link transforms, and the fixed transform from the final arm link to the hand. It is not usually one independent XYZ value you set each simulation step.

In Python, `EE_BODY_NAME = "hand"` chooses which MuJoCo body is used as the end-effector position. The function `_ee_pos()` reads its world position:

```python
return self.data.xpos[self.ee_body_id].copy()
```

If your XML uses another body name, update `EE_BODY_NAME`.

### Where is the gripper's open/closed behaviour defined?

Look in the XML's `<actuator>` section and the included Panda model. The gripper actuator/tendon configuration defines which joints it drives and its control range. This code assumes the standard convention:

- `0` = closed
- `255` = open

Verify this against your actual XML. If the gripper actuator has a different name or range, edit `GRIPPER_ACTUATOR_NAME`, `GRIPPER_OPEN_CTRL`, and `GRIPPER_CLOSE_CTRL` in `pick_lift_env.py`.

## 3. State / observation: where the 26 numbers come from

In `pick_lift_env.py`, look for:

```python
def _observation(self):
```

It concatenates:

| Slice | Values | Source |
|---|---|---|
| 0–6 | Seven joint positions | `data.qpos` at the seven arm joint addresses |
| 7–13 | Seven joint velocities | `data.qvel` at the seven arm DOF addresses |
| 14–16 | Hand XYZ | `data.xpos[hand_body_id]` |
| 17–19 | Cube XYZ | `data.xpos[cube_body_id]` |
| 20–22 | Target XYZ | `data.site_xpos[drop_spot_id]` |
| 23 | Gripper opening feature | Normalised current gripper command |
| 24 | Left finger force | Estimated from finger–cube contact |
| 25 | Right finger force | Estimated from finger–cube contact |

Total: 26 features.

This specific version is the original 24D vector (`7+7+3+3+3+1`) plus two force values. It does not additionally include cube-relative XYZ or cube velocity. Those were discussed as possible extra features, but adding them would change the dimension beyond 26.

## 4. How finger forces are estimated

Find:

```python
def _finger_forces(self):
```

The function:
1. Finds the cube's collision geoms.
2. Iterates over MuJoCo contacts in `data.contact`.
3. Keeps contacts where one geom belongs to the cube.
4. Checks whether the other geom's name matches the configured left/right finger substring.
5. Calls `mujoco.mj_contactForce(...)`.
6. Uses the magnitude of the contact force's first three components and sums it for that finger.

The configurable name fragments are near the top:

```python
LEFT_FINGER_KEY = "leftfinger"
RIGHT_FINGER_KEY = "rightfinger"
```

**You must check the actual geom names in your XML.** If the names differ, update these strings. Otherwise the measured force may stay zero. Also verify the force units and contact-frame convention for your model and MuJoCo version before treating these as calibrated sensor readings.

## 5. Action: where the eight commands are interpreted

Find:

```python
def _apply_action(self, action):
```

The action has eight values, each clipped to `[-1, 1]`.

### First seven values: incremental arm joint targets

For each joint:

```python
self.arm_targets = self.arm_targets + ARM_ACTION_SCALE_RAD * action[:7]
```

With `ARM_ACTION_SCALE_RAD = 0.035`, an action value of `+1` increases that joint's target by 0.035 radians; `-1` decreases it by 0.035 radians. The code clips the target to the joint's XML range when the joint is limited, then writes it to the matching actuator's `data.ctrl` slot.

This assumes the arm actuators behave like position-target actuators. If the XML uses torque actuators, this mapping must be changed.

### Eighth value: gripper command

```python
if action[7] > 0.2:
    self.gripper_command = GRIPPER_OPEN_CTRL
elif action[7] < -0.2:
    self.gripper_command = GRIPPER_CLOSE_CTRL
```

A positive command opens; a negative command closes; values between -0.2 and +0.2 keep the previous command. This avoids the gripper rapidly toggling around zero.

## 6. Where the reward function is defined

The reward is calculated in:

```python
def step(self, action):
```

This is the key Gymnasium loop. It records the pre-action state, applies the action, advances MuJoCo, reads the new state, and calculates reward.

### Reward terms

The constants are near the top of `pick_lift_env.py`, so you can change them without searching through the whole function.

| Code term | Meaning | Starting value |
|---|---|---:|
| `W_APPROACH_PROGRESS * (distance_before - distance_after)` | Rewards reduction in hand-to-cube distance | 5.0 |
| `-W_DISTANCE * distance_after` | Small penalty for remaining far away | 0.05 |
| `R_FIRST_FINGER_CONTACT` | One-time bonus on first finger contact | +1 |
| `R_BILATERAL_CONTACT` | One-time bonus when both fingers first contact | +2 |
| `W_LIFT_PROGRESS * lift_progress` | Rewards upward cube motion, only with bilateral contact | 20.0 |
| `-W_FORCE_TARGET * (grip_force - TARGET_GRIP_FORCE_N)**2` | Penalises deviation from target mean gripping force | weight 0.05; target 2 N |
| `-W_EXCESS_FORCE * excess**2` | Extra penalty above the force threshold | weight 0.1; threshold 8 N |
| `R_HOLD_PER_STEP` | Reward each action that maintains the required lift with bilateral contact | +0.5 |
| `R_SUCCESS` | One-time bonus after 10 consecutive successful hold steps | +25 |
| `R_DROP` | Penalty if the lifted cube drops or falls below the world-z threshold | -10 |

The constants are experimental starting points, not scientifically proven optimal values. The 2 N target and 8 N threshold are placeholders until you inspect real force magnitudes in your simulation.

### Why progress rewards use differences

For distance, the term is:

```python
distance_before - distance_after
```

If distance decreases, this is positive. If the robot moves farther away, it is negative.

For lift:

```python
max(0.0, lift_after - lift_before)
```

Only upward progress is rewarded. Downward motion is handled by hold/drop conditions rather than by a negative lift-progress term.

### Hold bonus and success condition

The cube must be at least `LIFT_SUCCESS_HEIGHT = 0.05` metres above its initial height and both fingers must register contact. `hold_counter` counts consecutive environment actions satisfying those conditions. If the count reaches `SUCCESS_HOLD_STEPS = 10`, the environment adds the one-time success bonus and terminates the episode.

This is a practical proxy for stable grasping, not a perfect physical proof that the fingers are carrying all the cube's weight. Inspect contact/force and cube-motion logs during debugging.

### Drop penalty

Once the cube has been lifted, if it falls back to less than 2 cm above its initial height, the episode ends with `R_DROP = -10`. It also ends with that penalty if the cube falls below `FALL_Z_THRESHOLD = 0.05` metres in world coordinates.

## 7. How one environment step advances MuJoCo

The environment follows this order:

1. Read the current hand and cube positions.
2. Apply the eight-dimensional action to actuator controls.
3. Call `mujoco.mj_step(model, data)` five times.
4. Read the updated hand/cube positions and finger forces.
5. Calculate reward, update the hold counter, and check success/drop conditions.
6. Return the new observation, reward, termination flags and diagnostic information.

Five physics steps are taken per policy action, controlled by `SIM_STEPS_PER_ACTION = 5`. The action is therefore held constant for those five MuJoCo integration steps.

## 8. How training uses this environment

In `train_ppo_cpu.py`, the `PandaLiftEnv` is wrapped in a `Monitor`, then passed to Stable-Baselines3 PPO. The script uses the simple initial PPO settings:

- 300,000 timesteps by default
- One environment by default
- Learning rate: `3e-4`
- Rollout length: `1024`
- Batch size: `64`
- Epochs: `10`
- Discount factor: `0.99`
- GAE lambda: `0.95`
- Entropy coefficient: `0.01`
- `device="cpu"`

The PPO code does not know the physical meaning of the XML directly. It receives the 26 numbers from `_observation()`, outputs eight action values, and uses the scalar reward returned by `step()` to learn.

## 9. What to inspect when results look wrong

- **Arm doesn't move:** check arm actuator types and whether their `ctrlrange` is compatible with joint-angle targets.
- **Gripper doesn't open/close:** verify actuator name, control range, and open/close direction in the XML.
- **Finger forces always zero:** print all geom names and change the left/right name fragments.
- **Robot gets close but doesn't grasp:** inspect gripper orientation, finger geometry, cube dimensions and contact forces.
- **Reward is dominated by force penalty:** log `left_finger_force_N`, `right_finger_force_N` and tune the target/weights after seeing actual values.
- **Success never occurs:** inspect the cube lift height, bilateral-contact flags, and hold counter.
- **Training is slow:** keep `n_envs=1` for the first CPU baseline; do not increase parallelism until the environment works reliably.

## 10. Suggested first experiment

1. Run a short environment smoke test before PPO training.
2. Confirm observation shape is `(26,)` and action shape is `(8,)`.
3. Confirm finger-force values become nonzero when each finger contacts the cube.
4. Confirm a manually selected action can move the arm and open/close the gripper.
5. Train for a short budget and inspect success rate and reward logs.
6. Change one reward term at a time and compare evaluation results.

The first goal is to validate the interface between the XML and Python. Only after that should reward-weight tuning be interpreted as a learning experiment.
