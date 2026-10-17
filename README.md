# Force-aware Panda Pick-and-Lift — CPU experiment

This folder contains a Gymnasium/MuJoCo environment and a simple PPO training script for a Franka Emika Panda robot to grasp a cube and hold it at least 5 cm above its initial height.

> **Important:** I cannot inspect your local XML from this chat. The code uses the names from our discussion (`cube`, `hand`, `drop_spot`, `joint1`–`joint7`) and configurable left/right finger geom-name fragments. Check those names in your actual XML before training.

## Files

- `pick_lift_env.py` — MuJoCo environment, observation/action definitions, contact-force extraction, reward, termination and success conditions.
- `train_ppo_cpu.py` — initial/simple PPO configuration, explicitly running on CPU.
- `MUJOCO_AND_CODE_WALKTHROUGH.md` — detailed walkthrough of XML, state, action, reward and the simulation loop.

## Install

Activate your project virtual environment, then install:

```powershell
python -m pip install mujoco gymnasium stable-baselines3 tensorboard
```

## Run

From the root of your project (the directory that contains `mujoco_menagerie`), copy the two `.py` files there and run:

```powershell
python train_ppo_cpu.py --xml "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml"
```

Optional:

```powershell
python train_ppo_cpu.py --xml "mujoco_menagerie/franka_emika_panda/panda_pick_place.xml" --timesteps 300000 --n-envs 1
```

The output model is saved under `models/panda_lift_ppo.zip`; logs go under `logs/panda_lift`.

## Before training: check these XML-specific names

1. Cube body is named `cube`.
2. End-effector body is named `hand`. If not, edit `EE_BODY_NAME`.
3. Target site is named `drop_spot`. The task is lift-only, but its XYZ remains in the observation for compatibility with the original 24D state.
4. Seven arm joints are named `joint1` through `joint7`.
5. Gripper actuator is named `actuator8`; if not, update `GRIPPER_ACTUATOR_NAME`.
6. Left/right finger geom names contain `leftfinger` and `rightfinger`. If not, change `LEFT_FINGER_KEY` and `RIGHT_FINGER_KEY`.

**Force warnings matter:** if the left/right geom fragments do not match, that force observation will stay at zero and the force-related reward terms will not work properly. Confirm the actual geom names in the XML before interpreting results.

## State dimension

The observation is deliberately **26D = the original 24D state plus two finger-force values**:

- 7 arm joint positions
- 7 arm joint velocities
- 3 end-effector position coordinates
- 3 cube position coordinates
- 3 target position coordinates
- 1 gripper opening feature
- 1 left finger force
- 1 right finger force

Total: `7 + 7 + 3 + 3 + 3 + 1 + 1 + 1 = 26`.

The cube-relative position is not included separately in this 26D version because it would make the vector larger than 26D. The gripper-to-cube distance is computed internally for the reward. If you want cube-relative XYZ and cube velocity in the observation as well, this becomes a different observation design and its dimension must be updated.

## Action dimension

The action is an 8D vector, each component clipped to [-1, 1]:

- Actions 0–6 increment the seven arm joint target angles by up to `0.035 rad` per environment action.
- Action 7 opens the gripper if above `+0.2`, closes it if below `-0.2`, and otherwise leaves the previous gripper command unchanged.

The script assumes position-style arm actuators and the usual Panda gripper command convention `0=closed`, `255=open`. Verify that this matches your XML.

## Reward summary

The environment uses:
- `5 * (distance_before - distance_after)`: reward for approach progress.
- `-0.05 * distance_after`: small penalty for remaining far from the cube.
- `+1` once on first finger contact.
- `+2` once when bilateral contact is first detected.
- `+20 * positive_lift_progress_in_metres`, only while both finger contacts are detected.
- A force-target penalty while bilateral contact exists: `-0.05 * (mean_finger_force - 2 N)^2`.
- An excessive-force penalty above 8 N: `-0.1 * max(0, mean_force - 8 N)^2`.
- `+0.5` per environment action while the cube is at least 5 cm up and both fingers still register contact.
- `+25` once when that stable hold lasts 10 consecutive environment actions.
- `-10` if the cube is dropped after lifting or falls below the world-z threshold.

All numerical weights and force thresholds are starting experimental values, not validated optima. The 2 N target and 8 N threshold are placeholders; inspect force magnitudes in your simulation and tune them. Because the force term is quadratic, force units and scale matter.

## Caveats

- This is a runnable starting point, not a guarantee that the local XML matches all names or actuator conventions.
- Contact forces are estimated from MuJoCo's per-contact force output and grouped by finger geometry names. The force coordinate frame and exact geom naming should be verified for the installed MuJoCo version and your model.
- The script does not use GPU/CUDA.
- `n_envs=1` is the simplest and most portable CPU baseline.
- The task is lift-only; the target site remains in the observation because the original state definition included target XYZ, but the reward does not use target distance.
