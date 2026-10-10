# Panda PPO starter (MuJoCo)

## Put these files in your existing `proj1` folder
- `pick_place_env.py`
- `train_ppo.py`
- `visualize_ppo.py`

Expected XML:
`mujoco_menagerie/franka_emika_panda/panda_pick_place.xml`

Expected XML names:
- body: `cube`
- site: `drop_spot`
- body: `hand`
- arm joints: `joint1` ... `joint7`
- one gripper actuator (preferably with "grip" in its name, otherwise actuator index 7)

The environment prefers an end-effector site named `attachment_site`; if absent, it uses the offset from the previous IK script.

## Install in your existing virtual environment
```powershell
python -m pip install mujoco gymnasium stable-baselines3 tensorboard
```

## Important: verify gripper control before training
The code assumes the latest diagnostic from your `ik.py`: gripper ctrl low = closed, ctrl high = open.
If your XML behaves the opposite way, swap `self.gripper_close_ctrl` and `self.gripper_open_ctrl` in `pick_place_env.py`.

## 1. Watch random actions (before training)
```powershell
python visualize_ppo.py --random --episodes 1
```
The arm will look chaotic. This is expected: these are random actions, not learning.

## 2. Train PPO
```powershell
python train_ppo.py
```
Training runs without a live viewer so simulation runs faster. Checkpoints are saved in `models/`.
Start with 300,000 steps; this is a smoke test, not a promise of success.

## 3. Watch the trained policy
```powershell
python visualize_ppo.py --model models/panda_ppo_final.zip --episodes 3
```
To slow it down, use `--delay 0.05`.

## 4. Watch training metrics
```powershell
tensorboard --logdir logs/tensorboard
```
Open the local URL printed by TensorBoard.

## Debug notes
- The cube must be a body named `cube` with a free joint.
- `drop_spot` should be a site at the intended target location.
- This starter assumes the table/cube geometry is already physically valid.
- The end-effector position is best taken from a site named `attachment_site`. If that site does not represent the midpoint between the fingers, change the site name or calibrate the fallback offset.
- PPO from random initialization may not learn reliable grasping with this initial reward. First confirm the viewer opens, the arm moves, the gripper opens/closes, and reward diagnostics make sense.
- This code is a starter scaffold; exact actuator semantics and XML names must match the local model.
