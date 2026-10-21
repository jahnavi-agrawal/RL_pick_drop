# Panda Cube Lift: PPO Experiments

This repository compares two CPU-based PPO experiments for a Franka Emika Panda in MuJoCo. Both use the same 26-dimensional observation, 8-dimensional action, XML scene, and IK-derived starting pose. The main comparison is the reward design:

1. **Experiment 1 — PPO baseline:** `train_ppo_cpu.py` with `pick_lift_env.py`.
2. **Experiment 2 — lift-focused reward:** `train_ppo_lift_focus.py`, which uses a separate environment subclass and leaves the baseline environment unchanged.

The robot starts at the IK-derived grasp configuration from `grasp_q.npy`, with the gripper open. This is already at the cube; it is not an approach-from-home task. Both experiments therefore disable approach-distance reward. The target position is retained in the observation for compatibility, but neither reward uses it.

## Setup and running

Activate the project environment from the repository root:

```bash
source robot_rl/bin/activate
```

Run Experiment 1:

```bash
python train_ppo_cpu.py
```

Run Experiment 2:

```bash
python train_ppo_lift_focus.py
```

Both scripts default to one environment and CPU training. Experiment 1 defaults to 1,500,000 timesteps; Experiment 2 defaults to 500,000 timesteps. Each writes models, configuration, and logs to its own output directory.

## 1. Shared state space (observation space)

Both experiments use the same 26-dimensional observation:

$$
\mathbf{s}_t=\mathbf{o}_t=
\begin{bmatrix}
\mathbf{q}_t\\
\dot{\mathbf{q}}_t\\
\mathbf{p}^{ee}_t\\
\mathbf{p}^{cube}_t\\
\mathbf{p}^{target}\\
g_t\\
f_t^L\\
f_t^R
\end{bmatrix}\in\mathbb{R}^{26}.
$$

Expanded form:

$$
\mathbf{s}_t=
\begin{bmatrix}
q_1,\ldots,q_7,
\dot q_1,\ldots,\dot q_7,
x_{ee},y_{ee},z_{ee},
x_c,y_c,z_c,
x_{tar},y_{tar},z_{tar},
g_t,f_t^L,f_t^R
\end{bmatrix}^{T}.
$$

| Variable | Dimension | Description |
|---|---:|---|
| $\mathbf{q}_t$ | 7 | Panda arm joint positions (rad) |
| $\dot{\mathbf{q}}_t$ | 7 | Panda arm joint velocities (rad/s) |
| $\mathbf{p}^{ee}_t$ | 3 | `hand` body position in world coordinates (m) |
| $\mathbf{p}^{cube}_t$ | 3 | Cube position in world coordinates (m) |
| $\mathbf{p}^{target}$ | 3 | `drop_spot` site position in world coordinates (m) |
| $g_t$ | 1 | Normalized gripper control feature; 0 is closed and 1 is open |
| $f_t^L$ | 1 | Estimated left-finger contact force (N) |
| $f_t^R$ | 1 | Estimated right-finger contact force (N) |

Total: $7+7+3+3+3+1+1+1=26$.

Finger forces are computed from MuJoCo contact forces between the cube and the left and right finger bodies. The target and forces are observed, but the target position is not part of either reward. Force values depend on the XML geometry, contact settings, and MuJoCo model.

## 2. Shared action space

Both experiments use an 8-dimensional action clipped to $[-1,1]^8$:

$$
\mathbf{a}_t=
\begin{bmatrix}
a_t^1 & a_t^2 & \cdots & a_t^7 & a_t^g
\end{bmatrix}^{T}\in[-1,1]^8.
$$

The first seven values increment the arm's joint targets:


$$q_{t+1}^{i,\mathrm{target}} = \text{clip}\left(q_t^{i,\mathrm{target}} + 0.035a_t^i, q_{\min}^i, q_{\max}^i\right), \qquad i = 1, \ldots, 7.$$


The increment scale is $0.035$ rad per environment action. The eighth value controls the gripper:

$$
u_t^g=
\begin{cases}
255, & a_t^g>0.2 \quad (\text{open}),\\
0, & a_t^g<-0.2 \quad (\text{close}),\\
u_{t-1}^g, & -0.2\le a_t^g\le0.2 \quad (\text{keep previous command}).
\end{cases}
$$

The scene uses the Panda convention $0=\text{closed}$ and $255=\text{open}$. One environment action advances the MuJoCo simulation by five physics steps.

## 3. Experiment 1 — PPO baseline

**Files:** `train_ppo_cpu.py` and `pick_lift_env.py`

$$
 r_t = \begin{cases} +5, & \text{approach cube},\\ +1, & \text{first finger contact},\\ +2, & \text{both fingers contact the cube},\\ +20\Delta h_t, & \text{cube lifted with bilateral contact},\\ -0.05(\bar F_t-2)^2, & \text{force regulation, bilateral contact},\\ -0.1\max(0,\bar F_t-8)^2, & \text{excessive force, bilateral contact},\\ +0.5, & \text{cube held at least 5 cm high},\\ +25, & \text{successful lift},\\ -10, & \text{cube dropped},\\ 0, & \text{otherwise}. \end{cases} 
$$

### Training defaults

| Setting | Default |
|---|---:|
| Total timesteps | 1,500,000 |
| Parallel environments | 1 |
| Device | CPU |
| Learning rate | $3\times10^{-4}$ |
| Rollout steps (`n_steps`) | 1,024 |
| Batch size | 64 |
| PPO epochs | 10 |
| Discount (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |


## 4. Experiment 2 — lift-focused reward

**File:** `train_ppo_lift_focus.py`

**Question:** Does a reward that focuses credit on bilateral contact, lift progress, and stable holding provide a clearer learning signal from the IK grasp start?

Experiment 2 uses the same state, actions, reset pose, task success condition, and PPO hyperparameters as Experiment 1. It subclasses the baseline environment and changes only the reward calculation and evaluation reporting.


$$
r_t =
\begin{cases}
+1, & \text{bilateral contact established},\\
+20\min(\Delta h_t,0.005), & \text{lift reward per meter, bilateral contact},\\
-0.05\max(0,\bar F_t-8)^2, & \text{excessive force penalty, bilateral contact},\\
+0.5, & \text {hold bonus per step, } h_t  \geq 0.05 \text{ m},\\
+25, & \text{successful lift and hold},\\
-10, & \text{cube dropped},\\
0, & \text{otherwise}.
\end{cases}
$$

### What changed and why

| Reward element | Experiment 1 | Experiment 2 | Reason for change |
|---|---|---|---|
| Approach progress/distance | Disabled | Disabled | The IK reset already puts the arm at the grasp pose; approach is not the behavior being learned. |
| First finger contact | $+1$ on transition from zero contacts | Removed | Credit is focused on establishing a two-finger grasp rather than touching with one finger. |
| Bilateral contact | $+2$ on transition to two contacts | $+1$ on transition to two contacts | Keeps a contact signal while making it less dominant relative to task completion. |
| Lift progress | $20B_t\Delta h_t$ | Same scale, with progress capped at 5 mm per action | Limits occasional large rewards from contact or physics impulses. |
| Force penalty | Target 2 N plus penalty above 8 N | Penalty only above 8 N | Avoids steering the policy toward an unvalidated target force. |
| Stable hold/success/drop | $+0.5$ per qualifying action, $+25$ success, $-10$ drop | Unchanged | Keeps the task outcome definition comparable across reward experiments. |

Experiment 2 defaults to 500,000 training timesteps, one environment, and CPU. It uses the same PPO settings as Experiment 1. During training it evaluates every 25,000 steps over five episodes. At the end it evaluates the best available model over 20 episodes and writes success rate, mean return, mean maximum lift, bilateral-contact step rate, and drop count to `evaluation_summary.json`.

## 5. Comparing results

Compare both experiments using the same XML, `grasp_q.npy`, random seed, and PPO settings where possible. Mean reward alone is not enough: the reward definitions differ, so a higher reward in one experiment does not directly mean better task performance.

Use these task metrics:

- Success rate: episodes that held the cube at least 5 cm high with bilateral contact for 10 consecutive actions.
- Mean maximum lift height.
- Bilateral-contact step rate.
- Drop count.
- Mean episode return, interpreted only within each reward design.

Experiment 2 writes its end-of-run task metrics to `runs/panda_lift_focus_v1/evaluation_summary.json` by default. Experiment 1's evaluation artifacts are saved under `runs/panda_grasp_lift_v1/`; its evaluation callback selects checkpoints by mean reward.

## 6. Visualizing the scene and policy

Watch the trained Experiment 2 policy:

```bash
mjpython watch_ppo_lift_focus.py
```

On macOS, use `mjpython` for MuJoCo viewer scripts. The policy viewer expects the Experiment 2 best checkpoint at `runs/panda_lift_focus_v1/best/best_model.zip` by default.

## Files

- `pick_lift_env.py` — shared MuJoCo dynamics, observations, actions, reset, and Experiment 1 reward.
- `train_ppo_cpu.py` — Experiment 1 training and optional hyperparameter sweep.
- `train_ppo_lift_focus.py` — Experiment 2 reward variant, PPO training, and final task-metric evaluation.
- `watch_ppo_lift_focus.py` — visualizes an Experiment 2 policy.

## Notes and limitations

- Reward weights and the 2 N / 8 N force values are experimental starting points, not validated optima.
- Force extraction depends on the XML body names `cube`, `left_finger`, and `right_finger` and on MuJoCo's contact-force calculation.
- Both experiments use a fixed cube pose and an IK-derived arm start pose. They do not test generalization across randomized object positions or robot starting configurations.
- The target position is included in the observation for compatibility but is not used by the current lift-only rewards.
