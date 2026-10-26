# ===========================================================================
# LunarLander RSSM — Reward Predictability Test (L1 experiment)
#
# Tests whether r_t can be predicted from the OBSERVED quantities (o, a) alone.
#   * 1-step regressor:  r_t = f(o_t,           a_t)
#   * 2-step regressor:  r_t = f(o_{t-1}, o_t,  a_t)
#
# The 1-step val R^2 measures how Markovian the reward is in the raw observations.
# The 2-step − 1-step gap measures how much extra information history adds.
#
# Hypothesis (CROF interpretation of ROF):
#   * LunarLander reward includes a shaping term that is the *difference* between
#     two potential functions of consecutive states (Ng-Harada-Russell shaping),
#     plus a contact bonus and crash penalty that depend on the velocity *just
#     before* contact.  The standard 8-D obs does not by itself contain
#     sufficient info to recompute reward from (o_t, a_t) — historical context
#     (o_{t-1}) should help.  Expected: 1-step val R^2 noticeably below 1.0,
#     and 2-step val R^2 noticeably higher.
#
# Compare with the Reacher version of this script in reacher_rssm/.
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------
def log(msg, log_path=None):
    print(msg, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# -----------------------------------------------------------------------------
# Dataset loading  (LunarLander: discrete actions, 4 classes)
# -----------------------------------------------------------------------------
NUM_ACTIONS = 4   # 0=noop, 1=fire-left, 2=fire-main, 3=fire-right


def load_npz(path):
    """Load a Lander NPZ and build (prev_obs, obs, next_obs, action_onehot,
    reward, valid_2step, dones)."""
    d = np.load(path)
    obs = d["obs"].astype(np.float32)            # (N, 8)
    next_obs = d["next_obs"].astype(np.float32)  # (N, 8)  state_{t+1}
    act_id = d["actions"].astype(np.int64)       # (N,)  — discrete IDs
    rew = d["rewards"].astype(np.float32)        # (N,)
    ep = d["ep_index"].astype(np.int64)          # (N,)
    step = d["step_index"].astype(np.int64)      # (N,)
    dones = d["dones"].astype(np.int64)          # (N,)
    n = obs.shape[0]
    assert act_id.ndim == 1, f"Lander actions should be (N,), got {act_id.shape}"

    if act_id.size and (act_id.max() >= NUM_ACTIONS or act_id.min() < 0):
        raise ValueError(f"Action IDs out of range [0, {NUM_ACTIONS-1}]: "
                         f"min={act_id.min()} max={act_id.max()}")

    # One-hot encode the discrete action.
    act = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    act[np.arange(n), act_id] = 1.0

    # Build prev_obs by shifting; mark step_index==0 (and episode boundaries) invalid.
    prev_obs = np.zeros_like(obs)
    prev_obs[1:] = obs[:-1]
    valid_2step = step >= 1
    same_ep = np.zeros(n, dtype=bool)
    same_ep[1:] = ep[1:] == ep[:-1]
    valid_2step = valid_2step & same_ep

    return obs, act, rew, prev_obs, valid_2step, dones, next_obs


def filter_terminals(obs, act, rew, prev_obs, valid_2step, dones, next_obs):
    """Drop transitions where dones==1.  These carry the +100 land / -100 crash bonuses
    that depend on Box2D's lander.awake flag (not in the observation), so they are the
    main source of unexplainable reward variance for Lander.  We also drop them when
    using --use_next_obs because next_obs at terminals is the reset state of the
    next episode, not a continuation of the current trajectory."""
    keep = dones == 0
    return (obs[keep], act[keep], rew[keep], prev_obs[keep],
            valid_2step[keep], dones[keep], next_obs[keep],
            int(keep.sum()), int(keep.size))


def encode_inputs(obs, act, prev_obs=None, next_obs=None):
    """Concatenate inputs.

    Layout (with --use_next_obs disabled):
      * 1-step:   [obs,                act]                        -> 12-D
      * 2-step:   [prev_obs, obs,      act]                        -> 20-D

    Layout (with --use_next_obs enabled, "L1c" formulation):
      * 1-step+next: [obs,            act, next_obs]               -> 20-D
      * 2-step+next: [prev_obs, obs,  act, next_obs]               -> 28-D
    """
    parts = []
    if prev_obs is not None:
        parts.append(prev_obs)
    parts.append(obs)
    parts.append(act)
    if next_obs is not None:
        parts.append(next_obs)
    return np.concatenate(parts, axis=1)


# -----------------------------------------------------------------------------
# MLP model
# -----------------------------------------------------------------------------
class RewardMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Train / evaluate one regressor
# -----------------------------------------------------------------------------
def train_mlp(X_tr, y_tr, X_va, y_va, *, epochs=80, batch_size=2048, lr=1e-3,
              hidden=256, depth=2, weight_decay=1e-5, patience=10, device="cpu",
              log_path=None, label="MLP"):
    device = torch.device(device)
    Xtr_t = torch.from_numpy(X_tr).float()
    ytr_t = torch.from_numpy(y_tr).float()
    Xva_t = torch.from_numpy(X_va).float().to(device)
    yva_t = torch.from_numpy(y_va).float().to(device)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = RewardMLP(input_dim=X_tr.shape[1], hidden=hidden, depth=depth).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_r2 = -np.inf
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_mse = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t).cpu().numpy()
            pred_tr = model(Xtr_t.to(device)).cpu().numpy()
        val_r2 = r2_score(y_va, pred_va)
        train_r2 = r2_score(y_tr, pred_tr)

        if val_r2 > best_val_r2 + 1e-6:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(f"      [{label}] epoch {epoch:3d}/{epochs}  train_mse={train_mse:.4f}  "
                f"train_R^2={train_r2:.4f}  val_R^2={val_r2:.4f}  best_val_R^2={best_val_r2:.4f}",
                log_path)

        if epochs_since_best >= patience:
            log(f"      [{label}] early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)", log_path)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t).cpu().numpy()
        pred_tr = model(Xtr_t.to(device)).cpu().numpy()
    final_train_r2 = r2_score(y_tr, pred_tr)
    final_val_r2 = r2_score(y_va, pred_va)
    return final_train_r2, final_val_r2, pred_va


def train_linear(X_tr, y_tr, X_va, y_va):
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    pred_va = reg.predict(X_va)
    return (r2_score(y_tr, reg.predict(X_tr)),
            r2_score(y_va, pred_va),
            pred_va)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LunarLander reward-predictability test (L1)")
    parser.add_argument("--train_dataset",
                        default="datasets/2026_02_08_lunarlanderV2/lunarlander_train_dataset.npz")
    parser.add_argument("--val_dataset",
                        default="datasets/2026_02_08_lunarlanderV2/lunarlander_val_dataset.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_path", default="reward_predictability_logs.txt")
    parser.add_argument("--filter_terminals", action="store_true",
                        help="Drop transitions where dones==1.  For Lander this isolates "
                             "the shaping+engine portion of reward (excludes +100/-100 "
                             "terminal bonuses that depend on the Box2D awake flag).")
    parser.add_argument("--use_next_obs", action="store_true",
                        help="Add next_obs (state_{t+1}) to the regressor inputs.  This "
                             "is the L1c formulation: r_t = shaping(o_{t+1}) - shaping(o_t) "
                             "- engine_cost(a_t), so providing all three quantities should "
                             "yield R^2 ~= 1 on filtered Lander.")
    parser.add_argument("--save_predictions", default=None,
                        help="Path to save val predictions and metadata as a .npz file.  "
                             "Used by downstream scripts (e.g. wm_reward_sweep.py) to "
                             "compare WM reward-head outputs against the L1c MLP baseline "
                             "without retraining.  Includes both 1-step and 2-step variants.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_path = args.log_path
    if log_path is not None and os.path.dirname(log_path) and not os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log("=" * 78, log_path)
    log(f"LunarLander Reward-Predictability Test (L1)  --  {datetime.now():%Y-%m-%d %H:%M:%S}", log_path)
    log("=" * 78, log_path)
    log(f"train_dataset = {args.train_dataset}", log_path)
    log(f"val_dataset   = {args.val_dataset}", log_path)
    log(f"device        = {args.device}", log_path)
    log(f"hidden        = {args.hidden}, depth = {args.depth}", log_path)
    log(f"epochs        = {args.epochs}, batch_size = {args.batch_size}, lr = {args.lr}", log_path)
    log(f"filter_terminals = {args.filter_terminals}", log_path)
    log(f"use_next_obs     = {args.use_next_obs}", log_path)

    obs_tr, act_tr, rew_tr, prev_tr, mask_tr, dones_tr, next_tr = load_npz(args.train_dataset)
    obs_va, act_va, rew_va, prev_va, mask_va, dones_va, next_va = load_npz(args.val_dataset)

    if args.filter_terminals:
        log("\n[FILTER] Dropping terminal transitions (dones==1)...", log_path)
        n_term_tr = int((dones_tr == 1).sum())
        n_term_va = int((dones_va == 1).sum())
        log(f"  train: removing {n_term_tr} terminal transitions out of {dones_tr.size}", log_path)
        log(f"  val:   removing {n_term_va} terminal transitions out of {dones_va.size}", log_path)
        if n_term_tr > 0:
            term_rew_tr = rew_tr[dones_tr == 1]
            log(f"  train terminal reward stats: mean={term_rew_tr.mean():.4f}  "
                f"std={term_rew_tr.std():.4f}  min={term_rew_tr.min():.4f}  "
                f"max={term_rew_tr.max():.4f}", log_path)
        (obs_tr, act_tr, rew_tr, prev_tr, mask_tr, dones_tr, next_tr,
         kept_tr, _) = filter_terminals(obs_tr, act_tr, rew_tr, prev_tr,
                                         mask_tr, dones_tr, next_tr)
        (obs_va, act_va, rew_va, prev_va, mask_va, dones_va, next_va,
         kept_va, _) = filter_terminals(obs_va, act_va, rew_va, prev_va,
                                         mask_va, dones_va, next_va)
        log(f"  kept: train={kept_tr}, val={kept_va}", log_path)

    log(f"\nDataset sizes:", log_path)
    log(f"  train: N={obs_tr.shape[0]}, obs_dim={obs_tr.shape[1]}, "
        f"action_onehot_dim={act_tr.shape[1]} (#classes={NUM_ACTIONS})", log_path)
    log(f"  val  : N={obs_va.shape[0]}", log_path)
    log(f"  train 2-step valid: {int(mask_tr.sum())} / {obs_tr.shape[0]}", log_path)
    log(f"  val   2-step valid: {int(mask_va.sum())} / {obs_va.shape[0]}", log_path)
    log(f"  reward stats (train): mean={rew_tr.mean():.4f}  std={rew_tr.std():.4f}  "
        f"min={rew_tr.min():.4f}  max={rew_tr.max():.4f}", log_path)
    log(f"  reward stats (val)  : mean={rew_va.mean():.4f}  std={rew_va.std():.4f}  "
        f"min={rew_va.min():.4f}  max={rew_va.max():.4f}", log_path)

    # ----------------------------------------------------------------------
    # 1-step regressors
    # ----------------------------------------------------------------------
    log("\n" + "-" * 78, log_path)
    if args.use_next_obs:
        log("1-step+next regressor   r_t = f(o_t, a_t, o_{t+1})   "
            "inputs: 20-D = obs(8) + act_onehot(4) + next_obs(8)", log_path)
    else:
        log("1-step regressor   r_t = f(o_t, a_t)   "
            "inputs: 12-D = obs(8) + act_onehot(4)", log_path)
    log("-" * 78, log_path)

    next1_tr = next_tr if args.use_next_obs else None
    next1_va = next_va if args.use_next_obs else None
    X1_tr = encode_inputs(obs_tr, act_tr, prev_obs=None, next_obs=next1_tr)
    X1_va = encode_inputs(obs_va, act_va, prev_obs=None, next_obs=next1_va)

    scaler1 = StandardScaler().fit(X1_tr)
    X1_tr_s = scaler1.transform(X1_tr).astype(np.float32)
    X1_va_s = scaler1.transform(X1_va).astype(np.float32)

    log("  Linear baseline:", log_path)
    lin1_tr_r2, lin1_va_r2, lin1_pred_va = train_linear(X1_tr_s, rew_tr, X1_va_s, rew_va)
    log(f"    train R^2 = {lin1_tr_r2:.4f}    val R^2 = {lin1_va_r2:.4f}", log_path)

    log("  MLP regressor:", log_path)
    mlp1_tr_r2, mlp1_va_r2, mlp1_pred_va = train_mlp(
        X1_tr_s, rew_tr, X1_va_s, rew_va,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
        depth=args.depth, weight_decay=args.weight_decay, patience=args.patience,
        device=args.device, log_path=log_path, label="1-step MLP",
    )
    log(f"    train R^2 = {mlp1_tr_r2:.4f}    val R^2 = {mlp1_va_r2:.4f}", log_path)

    # ----------------------------------------------------------------------
    # 2-step regressors  (uses only valid pairs where prev_obs is in same episode)
    # ----------------------------------------------------------------------
    log("\n" + "-" * 78, log_path)
    if args.use_next_obs:
        log("2-step+next regressor   r_t = f(o_{t-1}, o_t, a_t, o_{t+1})   "
            "inputs: 28-D = prev(8) + obs(8) + act(4) + next_obs(8)", log_path)
    else:
        log("2-step regressor   r_t = f(o_{t-1}, o_t, a_t)   "
            "inputs: 20-D = prev(8) + obs(8) + act(4)", log_path)
    log("-" * 78, log_path)

    next2_tr = next_tr[mask_tr] if args.use_next_obs else None
    next2_va = next_va[mask_va] if args.use_next_obs else None
    X2_tr = encode_inputs(obs_tr[mask_tr], act_tr[mask_tr],
                          prev_obs=prev_tr[mask_tr], next_obs=next2_tr)
    X2_va = encode_inputs(obs_va[mask_va], act_va[mask_va],
                          prev_obs=prev_va[mask_va], next_obs=next2_va)
    y2_tr = rew_tr[mask_tr]
    y2_va = rew_va[mask_va]

    scaler2 = StandardScaler().fit(X2_tr)
    X2_tr_s = scaler2.transform(X2_tr).astype(np.float32)
    X2_va_s = scaler2.transform(X2_va).astype(np.float32)

    log("  Linear baseline:", log_path)
    lin2_tr_r2, lin2_va_r2, lin2_pred_va = train_linear(X2_tr_s, y2_tr, X2_va_s, y2_va)
    log(f"    train R^2 = {lin2_tr_r2:.4f}    val R^2 = {lin2_va_r2:.4f}", log_path)

    log("  MLP regressor:", log_path)
    mlp2_tr_r2, mlp2_va_r2, mlp2_pred_va = train_mlp(
        X2_tr_s, y2_tr, X2_va_s, y2_va,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
        depth=args.depth, weight_decay=args.weight_decay, patience=args.patience,
        device=args.device, log_path=log_path, label="2-step MLP",
    )
    log(f"    train R^2 = {mlp2_tr_r2:.4f}    val R^2 = {mlp2_va_r2:.4f}", log_path)

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    log("\n" + "=" * 78, log_path)
    log("Summary  (LunarLander)", log_path)
    log("=" * 78, log_path)
    log(f"  1-step Linear  val R^2 = {lin1_va_r2:.4f}", log_path)
    log(f"  1-step MLP     val R^2 = {mlp1_va_r2:.4f}", log_path)
    log(f"  2-step Linear  val R^2 = {lin2_va_r2:.4f}", log_path)
    log(f"  2-step MLP     val R^2 = {mlp2_va_r2:.4f}", log_path)
    log(f"  delta(2-step - 1-step) MLP = {mlp2_va_r2 - mlp1_va_r2:+.4f}   "
        f"(positive = history adds info)", log_path)

    if mlp1_va_r2 > 0.95:
        msg = "  -> Reward is essentially Markovian in (o, a).  Latent history adds nothing."
    elif mlp2_va_r2 - mlp1_va_r2 > 0.05:
        msg = "  -> Reward depends on history; 2-step is strictly better."
    else:
        msg = "  -> Reward is partially observable from (o, a); history barely helps."
    log(msg, log_path)
    log("=" * 78, log_path)

    # ----------------------------------------------------------------------
    # Optional: save val predictions and config metadata for downstream comparison
    # (e.g. WM reward-head sweep in wm_reward_sweep.py).
    # ----------------------------------------------------------------------
    if args.save_predictions:
        save_path = args.save_predictions
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        np.savez(
            save_path,
            l1c_pred_val_1step=mlp1_pred_va.astype(np.float32),
            l1c_pred_val_2step=mlp2_pred_va.astype(np.float32),
            true_val_1step=rew_va.astype(np.float32),
            true_val_2step=y2_va.astype(np.float32),
            dones_val_1step=dones_va.astype(np.int64),
            mask_2step_to_1step=mask_va.astype(np.bool_),
            filter_terminals=np.array(args.filter_terminals, dtype=np.bool_),
            use_next_obs=np.array(args.use_next_obs, dtype=np.bool_),
            val_R2_mlp_1step=np.float64(mlp1_va_r2),
            val_R2_mlp_2step=np.float64(mlp2_va_r2),
            val_R2_lin_1step=np.float64(lin1_va_r2),
            val_R2_lin_2step=np.float64(lin2_va_r2),
        )
        log(f"\nSaved val predictions and metadata to {save_path}", log_path)


if __name__ == "__main__":
    main()
