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

$$
q_{t+1}^{i,\mathrm{target}}=
\operatorname{clip}\left(q_t^{i,\mathrm{target}}+0.035a_t^i,
q_{\min}^i,q_{\max}^i\right),\qquad i=1,\ldots,7.
$$

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

**Question:** How does PPO learn to close the gripper and lift the cube from the IK grasp start under the baseline contact, force, and lift reward?

### Reward variables

Let the left and right contact indicators be:

$$
C_t^L=\mathbb{1}[f_t^L>0.1],\qquad
C_t^R=\mathbb{1}[f_t^R>0.1],\qquad
B_t=C_t^LC_t^R.
$$

Let $n_t=C_t^L+C_t^R$ be the number of fingers in contact. Define cube lift above its reset height, positive lift progress, and mean finger force as:

$$
h_t=\max(0,z_t^{cube}-z_0^{cube}),\qquad
\Delta h_t=\max(0,h_t-h_{t-1}),\qquad
\bar f_t=\frac{f_t^L+f_t^R}{2}.
$$

### Reward components

First-finger contact reward:

$$
r_t^{first}=\begin{cases}
1, & n_{t-1}=0\text{ and }n_t\ge1,\\
0, & \text{otherwise}.
\end{cases}
$$

Bilateral-contact reward:

$$
r_t^{bilateral}=\begin{cases}
2, & B_t=1\text{ and }n_{t-1}<2,\\
0, & \text{otherwise}.
\end{cases}
$$

These transition rewards can be earned again if contact is lost and then re-established.

Lift-progress reward, only when both fingers contact the cube:

$$
r_t^{lift}=20B_t\Delta h_t.
$$

Force penalty, also applied only during bilateral contact:

$$
P_t^{force}=B_t\left[
0.05(\bar f_t-2)^2+
0.1\max(0,\bar f_t-8)^2
\right].
$$

The first term attracts the mean force toward a 2 N target; the second adds a penalty above 8 N. These values are initial tuning choices, not measured safe or optimal force targets.

Holding reward:

$$
r_t^{hold}=\begin{cases}
0.5, & h_t\ge0.05\text{ m and }B_t=1,\\
0, & \text{otherwise}.
\end{cases}
$$

Success reward is $25$ once the hold condition has remained true for 10 consecutive environment actions. Losing height or bilateral contact resets the consecutive-action counter.

Equivalently, with $H_t=\mathbb{1}[h_t\ge0.05\text{ m and }B_t=1]$:

$$
r_t^{success}=\begin{cases}
25, & H_t=H_{t-1}=\cdots=H_{t-9}=1\text{ and success has not been paid},\\
0, & \text{otherwise}.
\end{cases}
$$

Drop/fall reward:

$$
r_t^{drop}=\begin{cases}
-10, & \text{drop/fall termination occurs before success},\\
0, & \text{otherwise}.
\end{cases}
$$

A drop is triggered if the cube falls below $0.02$ m of lift after having been lifted, or if its world-frame height drops below $0.05$ m.

Total Experiment 1 reward:

$$
\boxed{
r_t^{(1)}=
r_t^{first}+r_t^{bilateral}+20B_t\Delta h_t
-P_t^{force}+r_t^{hold}+r_t^{success}+r_t^{drop}
}.
$$

**Approach reward status:** The approach-progress and distance weights exist in `pick_lift_env.py`, but their reward expressions are commented out. The implemented reward starts from zero and does not award approach progress or penalize distance. This matches the IK grasp starting pose.

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

The optional `--sweep` mode compares the baseline settings with a lower learning rate and a lower entropy coefficient, using the same Experiment 1 reward. The sweep runs sequentially. Set `--sweep-timesteps` explicitly to control the per-trial budget.

## 4. Experiment 2 — lift-focused reward

**File:** `train_ppo_lift_focus.py`

**Question:** Does a reward that focuses credit on bilateral contact, lift progress, and stable holding provide a clearer learning signal from the IK grasp start?

Experiment 2 uses the same state, actions, reset pose, task success condition, and PPO hyperparameters as Experiment 1. It subclasses the baseline environment and changes only the reward calculation and evaluation reporting.

### Reward components

The contact indicators, bilateral-contact indicator, lift height, and force definitions are the same as in Experiment 1.

One bilateral-contact transition bonus:

$$
r_t^{contact,2}=\begin{cases}
1, & B_t=1\text{ and }n_{t-1}<2,\\
0, & \text{otherwise}.
\end{cases}
$$

This bonus can recur after bilateral contact is lost and re-established. Experiment 2 removes the separate first-finger-contact bonus.

Capped lift-progress reward:

$$
r_t^{lift,2}=20B_t\min\left(\max(0,h_t-h_{t-1}),0.005\text{ m}\right).
$$

The per-action lift contribution is capped at $20\times0.005=0.1$, limiting reward spikes caused by sudden contact impulses.

Excessive-force penalty:

$$
P_t^{force,2}=0.05B_t\max(0,\bar f_t-8)^2.
$$

Experiment 2 removes the 2 N force-target penalty. It only penalizes mean finger force above 8 N, reducing pressure to seek a particular force value before actual force behavior has been calibrated.

The hold reward ($+0.5$ per qualifying action), success reward ($+25$ after 10 consecutive qualifying actions, paid once), and drop/fall penalty ($-10$) retain the Experiment 1 definitions.

Total Experiment 2 reward:

$$
\boxed{
r_t^{(2)}=
r_t^{contact,2}+r_t^{lift,2}-P_t^{force,2}
+r_t^{hold}+r_t^{success}+r_t^{drop}
}.
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

View the frozen Panda and cube scene:

```bash
mjpython view_static_scene.py
```

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
- `view_static_scene.py` — opens a frozen scene with the arm and cube.
- `MUJOCO_AND_CODE_WALKTHROUGH.md` — additional model and simulation notes.

## Notes and limitations

- Reward weights and the 2 N / 8 N force values are experimental starting points, not validated optima.
- Force extraction depends on the XML body names `cube`, `left_finger`, and `right_finger` and on MuJoCo's contact-force calculation.
- Both experiments use a fixed cube pose and an IK-derived arm start pose. They do not test generalization across randomized object positions or robot starting configurations.
- The target position is included in the observation for compatibility but is not used by the current lift-only rewards.
