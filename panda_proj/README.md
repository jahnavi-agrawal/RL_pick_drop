# Panda Pick-and-Place — Setup Guide

## 1. Clone the project

```powershell
git clone https://github.com/jahnavi-agrawal/RL_pick_drop.git
cd RL_pick_drop
```

## 2. Set up the Python environment

Create and activate a virtual environment:

```powershell
python -m venv robot_rl
.\robot_rl\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install mujoco gymnasium stable-baselines3 numpy scipy
```

## 3. Get the Franka Emika Panda model

The project uses the **Franka Emika Panda** robot model from the MuJoCo Menagerie repository.

If the Menagerie folder is not already present, clone it:

```powershell
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

The Panda model is located at:

```text
mujoco_menagerie/
└── franka_emika_panda/
    ├── panda.xml
    └── ...
```

## 4. Add the pick-and-place scene

Copy `panda_pick_place.xml` from the `panda_proj` folder into the `franka_emika_panda` folder.

The final structure should look like this:

```text
RL_pick_drop/
├── mujoco_menagerie/
│   └── franka_emika_panda/
│       ├── panda.xml
│       ├── panda_pick_place.xml
│       └── ...
├── ik.py
├── pick_lift_env.py
├── smoke_test.py
└── ...
```

**Important:** Keep `panda_pick_place.xml` in the same folder as `panda.xml`, because the scene XML includes the Panda model using a relative path.

## 5. Run the project

Run the IK script to test the robot's approach and grasp:

```powershell
python .\ik.py
```

Run the environment smoke test:

```powershell
python .\smoke_test.py
```

The smoke test should confirm that the environment loads, the observation and action dimensions are correct, and a simulation step completes.

## Notes

- Run commands from the project root directory.
- Keep the `mujoco_menagerie` folder and its franka_emika_panda files intact.
- If using Git submodules instead of cloning Menagerie separately, initialise them with:

```powershell
git submodule update --init --recursive
```
