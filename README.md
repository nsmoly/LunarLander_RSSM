# LunarLander Policy Trained via Latent World Model (similar to RSSM in DreamerV2-V3)

Implementation of the latent world model similar to RSSM (DreamerV2-V3) for training the policy for LunarLander-V3 gymnasium simulation by using model-based reinforcement learning that uses the pretrained latent world model for offline neural rollouts. The repo also has a zero-shot learning MPC (model predictive control) based policy that uses the trained world model for rollouts to compute optimal actions. I also added a model-free RL policy (actor-critic) for comparison. World-model-based approaches are much more data efficient: both MPC and world-model-based AC reach a strong policy using only the offline dataset (~181K real env transitions across 872 episodes), whereas the model-free RL policy needs ~8.0M real env transitions to reach its best policy — roughly **44× more environment interaction**. All three methods reach competitive performance on the real LunarLander environment (mean returns of $+168.7$ for MPC, $+129.4$ for WM-based AC, $+179.4$ for model-free AC). The repo includes a sweep script (`eval_rl.py`) for systematically evaluating actor-critic checkpoints. A separate validation-time metrics suite — used to select world-model checkpoints without ever touching the real environment — is described in the accompanying paper (forthcoming) and will be released alongside it.

![Moonlander WorldModel Based Zero-shot MPC Policy](moonlander_mpc_1.jpg)

## Workflow

**World-model-based path** (model-based RL):

1. **Collect Dataset** → 2. **Train World Model** → 3. **Test World Model** → 4. **Train Actor-Critic (WM-based)** → 5. **Test Policy**

**Model-free path** (baseline comparison):

1. **Train Model-Free Actor-Critic** → 2. **Test Policy**

**MPC path** (no actor needed):

1. **Collect Dataset** → 2. **Train World Model** → 3. **WM MPC Policy (CEM)**

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

## 5. Train Actor-Critic (WM-based)

Train the policy (actor) and value function (critic) via imagined rollouts in the world model's latent space. Requires a trained world model. This is the model-based RL approach.

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

## 6. Test Policy

Run a trained policy in the LunarLander environment. Supports both WM-based (latent) and model-free (obs) actors. Runs **headless by default** for fast evaluation; pass `--render` to open a Pygame window with a stats overlay (episode, return, action probs, entropy, FPS). At the end of the run, a scorecard summary is printed (mean/worst return, perfect/negative/catastrophic counts, avg entropy, avg steps).

```bash
# Test WM-based actor headless (requires world model + latent actor)
python test_policy.py --actor_type latent --world_model world_model.pt --actor actor.pt --episodes 20

# Test model-free actor headless (no world model needed)
python test_policy.py --actor_type obs --actor actor_mf.pt --episodes 20

# Watch with rendering (window with overlay; press R to toggle, Q/ESC to quit)
python test_policy.py --actor_type latent --world_model world_model.pt --actor actor.pt --render --render_fps 30

# Stochastic action sampling instead of deterministic argmax
python test_policy.py --actor_type obs --actor actor_mf.pt --stochastic
```

| Option | Default | Description |
|--------|---------|-------------|
| `--actor_type` | `latent` | `latent` (WM-based, uses RSSM) or `obs` (model-free, raw observations) |
| `--config` | `config.yaml` | Config for model dimensions |
| `--world_model` | `world_model.pt` | World model checkpoint (only for `latent` actor) |
| `--actor` | `actor.pt` / `actor_mf.pt` | Actor checkpoint (default depends on `--actor_type`) |
| `--episodes` | 20 | Number of episodes to run |
| `--max_steps` | 600 | Max steps per episode |
| `--render` | off | Open a Pygame window with stats overlay during rollout |
| `--render_fps` | 30.0 | Visual FPS cap when rendering (sleep-based throttle) |
| `--deterministic` | *(default)* | Use argmax action selection |
| `--stochastic` | | Sample actions from the policy distribution |
| `--seed` | 12345 | Random seed for reproducibility |

Render-window keyboard controls: `R` toggles rendering on/off, `Q`/`ESC` quits.

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

## 8. Train Model-Free Actor-Critic (Baseline)

Train a policy directly from on-policy environment interactions without a world model. Uses the same A2C algorithm (GAE, entropy regularization) as the WM-based trainer, but operates on raw observations instead of latent states. Serves as a baseline for comparing sample efficiency and compute cost against the world-model-based approach.

```bash
# Train from scratch
python train_modelfree_actorcritic.py --seed 12345

# Resume from latest checkpoint
python train_modelfree_actorcritic.py --resume --seed 12345

# Train with rendering (slower, shows one episode per epoch)
python train_modelfree_actorcritic.py --render --seed 12345
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 1000 | Number of training epochs |
| `--episodes_per_epoch` | 50 | On-policy episodes collected per epoch |
| `--max_steps` | 600 | Max steps per episode |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--gamma` | 0.99 | Discount factor |
| `--lambda_gae` | 0.95 | GAE lambda |
| `--entropy_coeff` | 0.2 | Initial entropy coefficient |
| `--entropy_coeff_end` | 0.01 | Final entropy coefficient (linearly decayed) |
| `--hidden_dim` | 256 | Hidden dim for ActorObs / CriticObs |
| `--checkpoint_freq` | 10 | Save checkpoints every N epochs |
| `--resume` | | Resume from latest `actor_mf` / `critic_mf` checkpoint pair |
| `--render` | | Render one episode per epoch |
| `--seed` | 12345 | Random seed |

Checkpoints are saved as `actor_mf_<date>_<time>_epoch_<N>.pt` and `critic_mf_<date>_<time>_epoch_<N>.pt`. Training logs are written to `train_modelfree_actorcritic_logs.txt`.

---

## 9. Sweep Actor-Critic Checkpoints (`eval_rl.py`)

Run `test_policy.py` as a subprocess for every actor checkpoint in a folder and aggregate the results into a single log. Supports both WM-based AC (`--actor_type latent`, requires `--world_model`) and model-free AC (`--actor_type obs`).

```bash
# WM-based AC sweep (CROF-selected world model held fixed)
python eval_rl.py \
    --checkpoints_dir checkpoints \
    --actor_type latent \
    --world_model world_model.pt \
    --output rl_eval_logs.txt

# Model-free AC sweep (no world model)
python eval_rl.py \
    --checkpoints_dir checkpoints \
    --actor_type obs \
    --epoch_stride 5 \
    --output rl_eval_logs_mfAC.txt
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoints_dir` | *(required)* | Folder containing actor checkpoints |
| `--actor_type` | *(required)* | `latent` (WM-based) or `obs` (model-free) |
| `--world_model` | None | World model checkpoint (required when `--actor_type latent`) |
| `--actor_pattern` | `actor.*epoch_(\d+)\.pt$` | Regex matching actor filenames; group 1 must be the integer epoch |
| `--episodes` | 20 | Episodes per checkpoint |
| `--seed` | 12345 | Random seed (passed through to `test_policy.py`) |
| `--max_steps` | 600 | Max steps per episode (matches MPC sweep) |
| `--epoch_min` | None | Skip checkpoints below this epoch |
| `--epoch_max` | None | Skip checkpoints above this epoch |
| `--epoch_stride` | 1 | Take every N-th checkpoint (use 5 for model-free actors saved every 10 epochs) |
| `--output` | `rl_eval_logs.txt` | Output log file |
| `--config` | `config.yaml` | Path to config |
| `--append` | off | Append to output instead of overwriting |
| `--top_k` | 10 | Print top-K checkpoints by mean return at the end |

Runs are deterministic at the given seed. The sweep produces:
- a per-checkpoint scorecard block (full `test_policy.py` stdout) for each actor
- a sweep summary table (sorted by epoch) listing mean/worst return, perfect/negative/catastrophic counts, avg steps, avg entropy
- a top-K table sorted by mean return
- a `Best by mean_return` line identifying the winning checkpoint

This is the script used in the paper to compare WM-based AC (trained with WM 295), WM-based AC (trained with WM 200), and the model-free AC baseline.

---

## Sample Files (Checked In)

The repository includes sample datasets and trained checkpoints:

| File | Description |
|------|--------------|
| `lunarlander_train_dataset.npz` | Training dataset (750 episodes, 158,685 transitions) |
| `lunarlander_val_dataset.npz` | Validation dataset (122 episodes, 22,231 transitions) |
| `world_model.pt` | Pretrained world model checkpoint — **epoch 295** (CROF-selected safe peak; MPC mean +168.65, worst-case +3.82) |
| `actor.pt` | WM-based actor checkpoint — **epoch 200** (peak deterministic mean return +129.42 over 20 episodes) |
| `critic.pt` | WM-based critic checkpoint — **epoch 200** (paired with the actor above) |
| `actor_mf.pt` | Model-free actor checkpoint — **epoch 610** (peak deterministic mean return +179.44 over 20 episodes) |
| `critic_mf.pt` | Model-free critic checkpoint — **epoch 610** (paired with the actor above) |

Use `--checkpoint world_model.pt` when testing the world model.


## Quick Reference

```bash
# World-model-based pipeline
python collect_dataset.py --dataset lunarlander_train_dataset.npz --seed 12345
python replay_dataset.py --dataset lunarlander_train_dataset.npz --episodes 5

python train_models.py --phase world_model --train_dataset lunarlander_train_dataset.npz --val_dataset lunarlander_val_dataset.npz
python test_worldmodel.py --mode open --max_episodes 2 --plot_dir plots

python train_models.py --phase actor_critic --train_dataset lunarlander_train_dataset.npz
python test_policy.py --actor_type latent --world_model world_model.pt --actor actor.pt --episodes 20

# MPC (no actor needed, uses world model directly)
python wm_mpc_policy.py --config config.yaml --world_model world_model.pt --render --episodes 20

# Model-free baseline
python train_modelfree_actorcritic.py --seed 12345
python test_policy.py --actor_type obs --actor actor_mf.pt --episodes 20

# Sweep all actor-critic checkpoints in a folder (WM-based or model-free)
python eval_rl.py --checkpoints_dir checkpoints --actor_type latent --world_model world_model.pt --output rl_eval_logs.txt
python eval_rl.py --checkpoints_dir checkpoints --actor_type obs --epoch_stride 5 --output rl_eval_logs_mfAC.txt
```

---

## Notes on important details that make this version working

- **Decoder design to multi-head outputs**:
  - Shared decoder backbone from `[h, z]`
  - Physics head (6 continuous dims: `x, y, vx, vy, angle, ang_vel`)
  - Contact head (2 binary dims: leg contacts)
  - Done head (1 binary dim)
- **Loss design to match output types**:
  - Physics: MSE
  - Contact: BCE-with-logits
  - Done: BCE-with-logits
  - (plus existing reward and KL terms)
- **Model design**:
  - `Linear -> LayerNorm -> SiLU` are good setup for physics modelling
- **Posterior design**:
  - Posterior uses `obs_t` only, while `done` is a training target (predicted).

### Why these choices matter

- The main gain comes from **separating continuous and binary prediction heads/losses**, which reduced blurry terminal/contact dynamics and improved landing quality under MPC optimizer.
- Capacity, normalization (LinearNorm) and activation (SiLU) changes further improve optimization stability. 

### Checkpoint selection notes

- Standard training-time metrics (validation loss, reward RMSE, multi-step open-loop RMSE) all keep improving monotonically past the point of best MPC performance, and pick deeply overfit checkpoints (epoch ≥460) where MPC closed-loop return has collapsed. MAE-based variants are even less informative because they suppress the rare large prediction errors that matter for control.
- A separate set of validation-time *structural* metrics — Jacobian-based Reward Observability Fraction averaged over curated "good" and "bad" states, plus controllability/observability rank fractions and multi-step observation RMSE, combined into a composite score (CROF) — turn out to be the strongest predictors of MPC return (best Spearman ρ ≈ −0.71 in our sweep) and select checkpoints inside the high-MPC plateau without ever touching the real environment. Epoch 295 was selected via this procedure.
- The full analysis (definitions, sweeps, and selection results) will be published with the accompanying paper. The `world_model.pt` checkpoint shipped here is the one selected by that procedure.

### Checkpoints in the repo

- **World model** (`world_model.pt`): **epoch 295**, the CROF-selected "safe peak" — highest mean MPC return inside the high-quality plateau ($+168.65$) and the only checkpoint in the plateau with a non-negative worst-case episode ($+3.82$).
- **WM-based AC** (`actor.pt` + `critic.pt`): **epoch 200**, the peak of the AC sweep when trained on world model epoch 295. Real-env mean return $+129.42$ over 20 deterministic episodes (8/20 perfect landings).
- **Model-free AC** (`actor_mf.pt` + `critic_mf.pt`): **epoch 610**, the peak of the model-free sweep. Real-env mean return $+179.44$ over 20 deterministic episodes (12/20 perfect landings, 0 catastrophic), trained on $\approx 8.0$M real environment transitions.
- Older actor/critic checkpoints (e.g., `actor_ep750.pt`, `critic_ep750.pt`) are kept for reproducibility of earlier experiments. Full checkpoint sweeps live under `archive/`.
