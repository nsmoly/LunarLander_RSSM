# MoonLander Policy Trained via RSSM World Model (aka DreamerV2-V3)

Implementation of the latent world model similar to RSSM (DreamerV2-V3) for training the policy for LunarLander RL gym simulation by using model-based reinforcement learning that uses the pretrained latent world model for offline neural rollouts.

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

Config: `horizon`, `past_horizon`, `future_horizon` (must satisfy P+F=H), `batch_size`, `lr`, `epochs`, `auxiliary_rewards`, etc.

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

## Sample Files (Checked In)

The repository includes sample datasets and a trained world model checkpoint in the root folder:

| File | Description |
|------|--------------|
| `lunarlander_train_dataset.npz` | Sample training dataset |
| `lunarlander_val_dataset.npz` | Sample validation dataset |
| `world_model_good.pt` | Pretrained world model checkpoint |

Use `--checkpoint world_model_good.pt` when testing the world model.


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
