# PPO Training Notes — Franka Emika Panda Pick-and-Place

## 1. Project objective

Train a Proximal Policy Optimization (PPO) agent in MuJoCo to grasp a cube using the Franka Emika Panda robot, lift it at least 5 cm, and maintain the lift for 10 consecutive environment steps.

**Success criteria**
- The cube is grasped with both fingers.
- The cube rises at least 0.05 m above its initial height.
- The lift is maintained for 10 consecutive environment steps.
- The cube does not fall or get knocked away.

## 2. Environment and baseline

| Component | Configuration |
|---|---|
| Simulator | MuJoCo |
| Robot | Franka Emika Panda |
| RL algorithm | PPO |
| Policy | MlpPolicy |
| Observation | 26-dimensional vector |
| Action | 8-dimensional vector |
| Compute | CPU |
| Baseline | Scripted inverse kinematics (IK) |

The IK script establishes a working approach and grasp pose. PPO is intended to learn the grasp-and-lift behaviour from the configured starting state.

**Starting-state configuration:** Record whether training begins from the home position or the saved pre-grasp pose. If pre-grasp initialization is used, PPO is learning a narrower task than full pick-and-place from home.

## 3. PPO hyperparameters

Initial configuration:

| Parameter | Value |
|---|---:|
| Total timesteps | 100,000 |
| Learning rate | 0.0003 |
| Rollout steps (`n_steps`) | 1,024 |
| Batch size | 64 |
| Training epochs (`n_epochs`) | 10 |
| Discount factor (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient (`ent_coef`) | 0.01 |
| Policy | MlpPolicy |
| Parallel environments | 1 |
| Device | CPU |

These are starting values, not proven optimal parameters. Record the actual values used in every run.

## 4. Observation and action design

### Observation (26 values)

- Arm joint positions: 7
- Arm joint velocities: 7
- End-effector position: 3
- Cube position: 3
- Target position: 3
- Gripper state: 1
- Left and right finger contact forces: 2

### Action (8 values)

- First 7 actions: incremental arm joint targets.
- Final action: gripper open/close/hold command.

Current arm action scale: 0.035 radians per action unit, subject to clipping and actuator-control implementation.

## 5. Reward design

The current environment rewards:
- Progress toward the cube.
- Initial finger contact and bilateral contact.
- Incremental lifting while both fingers are in contact.
- Maintaining a successful lift.
- Completing the success condition.

It penalises:
- Distance from the cube.
- Excessive contact force.
- Dropping the cube.

The current implementation includes a one-time success bonus of +25, a drop penalty of -10, and a hold reward of +0.5 per qualifying step. Record any changes to these values.

## 6. Experiment log

Copy this table for each training run.

| Field | Run 1 |
|---|---|
| Date | |
| Starting state | |
| Total timesteps | |
| Learning rate | |
| Action scale | |
| Reward changes | |
| Random seed | |
| Training duration | |
| Mean episode reward | |
| Evaluation success rate | |
| Mean maximum lift height | |
| Notes | |

## 7. Model outputs

Keep training artefacts organised:

```text
runs/
├── run_001/
│   ├── best_model.zip
│   ├── final_model.zip
│   ├── vecnormalize.pkl   # Only if normalization is used
│   ├── evaluations.npz
│   ├── monitor.csv
│   └── config.txt
```

Only include files actually produced by the training script. Do not assume a best-model checkpoint exists unless evaluation callbacks are configured.

## 8. Evaluation protocol

Evaluate the saved policy in a separate environment, without updating its weights.

Report:
- Number of evaluation episodes.
- Number and percentage of successful episodes.
- Mean and standard deviation of episode reward.
- Mean maximum cube lift height.
- Percentage of episodes with bilateral finger contact.
- Number of dropped cubes.
- Whether success is sustained for 10 consecutive steps.

Use a fixed set of evaluation seeds and the same success definition when comparing runs.

## 9. Performance improvement plan

1. Verify that the reset state and action controls are correct.
2. Check whether the agent reaches the cube and achieves bilateral contact.
3. If contact occurs but lifting fails, inspect gripper control, contact forces, and lift rewards.
4. If progress is consistent but success remains low, increase training timesteps.
5. Tune one hyperparameter at a time and compare evaluation results.
6. Test the final policy over multiple unseen initial conditions before reporting generalisation.

## 10. Reproducibility

For every run, save:
- Hyperparameters and random seed.
- Exact code version or Git commit.
- Trained model checkpoint.
- Training logs and evaluation results.
- Any changes to the XML, reward function, observation space, or action space.

A successful training run is one in which the saved policy repeatedly meets the defined success criteria during independent evaluation—not merely one with increasing reward.