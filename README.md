# LunarLander Policy Trained via latent RSSM World Model (as in DreamerV2-V3)

Implementation of the latent world model similar to RSSM (as in DreamerV2-V3) for training the policy for LunarLander RL gym simulation by using model-based reinforcement learning that uses the pretrained latent world model for offline neural rollouts.

![Moonlander WorldModel Based Zero-shot MPC Policy](moonlander_mpc_1.jpg)

## Workflow

1. **Collect** → 2. **Train World Model** → 3. **Test World Model** → 4. **Train Actor-Critic** → 5. **Test Policy**

---

## 1. Collect Dataset

Collect human demonstrations or random episodes using keyboard control. Controls: LEFT/RIGHT/UP/DOWN arrows; ESC to end and save.

```bash
# Basic usage (saves to lunarlander_dataset.npz)
python collect_dataset.py

# Specify output path
python collect_dataset.py --dataset lunarlander_train_dataset.npz

# With custom seed for reproducibility
python collect_dataset.py --dataset lunarlander_train_dataset.npz --seed 12345
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `lunarlander_dataset.npz` | Output dataset .npz path |
| `--seed` | 0 | Base RNG seed for episode surfaces |

---

## 2. Replay Dataset

Replay recorded episodes in the LunarLander environment to verify the dataset.

```bash
# Replay 20 episodes (default)
python replay_dataset.py

# Replay with custom dataset and settings
python replay_dataset.py --dataset lunarlander_train_dataset.npz --episodes 50 --seed 12345

# Slower replay (0.1 seconds per step)
python replay_dataset.py --dataset lunarlander_train_dataset.npz --episodes 10 --sleep 0.1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `lunarlander_dataset.npz` | Path to dataset .npz |
| `--episodes` | 20 | Number of episodes to replay |
| `--seed` | 0 | Random seed for selecting episodes |
| `--sleep` | 0.02 | Sleep between steps (seconds) |

---

## 3. Train World Model

Train the RSSM world model on sequences from real episodes. Uses `SequenceDataset` with teacher forcing.

```bash
# Basic usage (reads config.yaml, default datasets)
python train_models.py --phase world_model

# With explicit dataset paths
python train_models.py --phase world_model --config config.yaml \
    --train_dataset lunarlander_train_dataset.npz \
    --val_dataset lunarlander_val_dataset.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--phase` | *(required)* | `world_model` or `actor_critic` |
| `--config` | `config.yaml` | Path to config file |
| `--train_dataset` | `lunarlander_train_dataset.npz` | Training dataset path |
| `--val_dataset` | `lunarlander_val_dataset.npz` | Validation dataset path |
| `--seed` | 12345 | Random seed for reproducibility |

Config is in `config.yaml`: `sequence_length`, `batch_size`, `lr`, `epochs`, etc.

---

## 4. Test World Model

Evaluate the trained world model. Modes: **teacher** (posterior reconstruction), **open** (prior rollout), **sim** (constant action rollout).

```bash
# Teacher mode (default): posterior reconstruction with GT actions
python test_worldmodel.py --mode teacher --max_episodes 5

# Open mode: prior rollout from first obs, compare to GT
python test_worldmodel.py --mode open --dataset lunarlander_val_dataset.npz --max_episodes 10

# Save plots and animations
python test_worldmodel.py --mode open --max_episodes 5 --plot_dir plots --animate

# Sim mode: constant action rollout
python test_worldmodel.py --mode sim --constant_action 0 --max_episodes 3

# Use specific checkpoint
python test_worldmodel.py --mode teacher --checkpoint world_model.pt --max_episodes 5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `lunarlander_val_dataset.npz` | Validation dataset path |
| `--config` | `config.yaml` | Config file |
| `--checkpoint` | `world_model.pt` | World model checkpoint path |
| `--mode` | `teacher` | `teacher` / `open` / `sim` |
| `--max_episodes` | 5 | Episodes to evaluate |
| `--seed` | 0 | Seed for episode selection |
| `--plot_dir` | None | Save plots to directory |
| `--animate` | False | Save animations (.mp4) |
| `--constant_action` | 0 | Action id for sim mode |

---

## 5. Train Actor-Critic

Train the policy (actor) and value function (critic) via imagined rollouts in latent space. Requires a trained world model.

```bash
# Basic usage
python train_models.py --phase actor_critic

# With explicit dataset path
python train_models.py --phase actor_critic --config config.yaml \
    --train_dataset lunarlander_train_dataset.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--phase` | *(required)* | `actor_critic` |
| `--config` | `config.yaml` | Path to config file |
| `--train_dataset` | `lunarlander_train_dataset.npz` | Training dataset path |

Config: `horizon`, `past_horizon`, `future_horizon` (must satisfy P+F=H), `batch_size`, `lr`, `epochs`, etc.

---

## 6. Test Actor-Critic (Policy)

Run the trained policy in the LunarLander environment. Loads world model and actor from config-matched checkpoints.

```bash
# Default: world_model.pt (or world_model_good.pt), actor.pt
python test_policy.py

# Specify checkpoints
python test_policy.py --world_model world_model_good.pt --actor actor.pt

# Use config for model dimensions
python test_policy.py --config config.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config.yaml` | Config for latent_dim, hidden_dim, action_dim |
| `--world_model` | `world_model.pt` | World model checkpoint (fallback: world_model_good.pt) |
| `--actor` | `actor.pt` | Actor checkpoint |

---

## 7. WM MPC Policy (CEM Planner)

Run model-predictive control directly in Gymnasium using the world model only (no actor network).
At each step it plans over discrete action sequences with CEM, executes the best first action, then replans.

```bash
# Basic run (with render)
python wm_mpc_policy.py --config config.yaml --world_model world_model.pt --render

# Faster/no-render run
python wm_mpc_policy.py --config config.yaml --world_model world_model.pt --episodes 10
```

Keyboard controls (render window):
- `R`: toggle render on/off
- `Q` or `ESC`: quit

Useful knobs:
- `--horizon`, `--population`, `--elites`, `--cem_iters` for planner strength/speed
- `--w_angle`, `--w_ang_vel`, `--w_vx`, `--w_vy_down` for landing objective weights

---

## Sample Files (Checked In)

The repository includes sample datasets and a trained world model and actor-critic checkpoints in the root folder:

| File | Description |
|------|--------------|
| `lunarlander_train_dataset.npz` | Sample training dataset |
| `lunarlander_val_dataset.npz` | Sample validation dataset |
| `world_model.pt` | Pretrained world model checkpoint (140 epoch) |
| `actor.pt` | Pretrained world model checkpoint (130 epoch) |
| `critic.pt` | Pretrained world model checkpoint (130 epoch) |

Use `--checkpoint world_model.pt` when testing the world model.


## Quick Reference

```bash
# Full pipeline
python collect_dataset.py --dataset lunarlander_train_dataset.npz --seed 12345
python replay_dataset.py --dataset lunarlander_train_dataset.npz --episodes 5

python train_models.py --phase world_model --train_dataset lunarlander_train_dataset.npz --val_dataset lunarlander_val_dataset.npz
python test_worldmodel.py --mode open --max_episodes 2 --plot_dir plots

python train_models.py --phase actor_critic --train_dataset lunarlander_train_dataset.npz
python test_policy.py
```

---

## Release Notes for V3 Version of WorldModel/RSSM in this Repo

This release introduced a focused world-model refactor that significantly improved closed-loop MPC behavior in `wm_mpc_policy.py`, especially near touchdown.

### What changed

- **Decoder redesign to multi-head outputs**:
  - Shared decoder backbone from `[h, z]`
  - Physics head (6 continuous dims: `x, y, vx, vy, angle, ang_vel`)
  - Contact head (2 binary dims: leg contacts)
  - Done head (1 binary dim)
- **Loss redesign to match output types**:
  - Physics: MSE
  - Contact: BCE-with-logits
  - Done: BCE-with-logits
  - (plus existing reward and KL terms)
- **Model capacity/activation updates**:
  - `latent_dim=16`, `hidden_dim=256`, `mlp_hidden_dim=256`
  - `Linear -> LayerNorm -> SiLU` in all key world-model MLP blocks
- **Posterior input decision finalized**:
  - Posterior uses `obs_t` only, while `done` is a training target (predicted).

### Why these changes mattered

- The main gain came from **separating continuous and binary prediction heads/losses**, which reduced blurry terminal/contact dynamics and improved landing quality under MPC optimizer.
- Capacity changes (reduced capacity) and normalization (LinearNorm) / activation (SiLU) changes further improved optimization stability, but the decoder/loss split was probably the biggest contributor. The training now converges to a good working world model by epoch 10. No dataset expansion was needed.

### Checkpoint selection notes

- In training logs, the most informative metrics for MPC checkpoint ranking were:
  - **validation loss (`val_loss`)**
  - **reward RMSE (`reward_rmse`)**
- Metrics like best late-stage observation RMSE alone were less predictive of MPC closed-loop quality.
MAE metrics were not that informative since they don't account for outliers as well as RMSE metrics.

### Checkpoints: working world_model.pt, actor.pt and critic.pt checkpoints are in the repo

- WorldModel checkpoint (world_model.pt) is for epoch 200
- AC RL Policy checkpoints (actor.pt and critic.pt) are for epoch 750
- Also the repo has a world_model_random.pt checkpoint with random weights to compare with and poorly trained AC RL policy in actor_earlyBad_ep50.pt
