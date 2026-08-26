# ===========================================================================
# MoonLander RSSM — Dreamer-style World Models for LunarLander Policy Training
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================

# Behaviour cloning baseline: train ActorObs to imitate successful human
# landings, with no world model, critic, or reward signal involved.
#
# Exists to measure how many demonstrations imitation needs to match what the
# world model reaches through MPC, so the checkpoint is deliberately written in
# the same format train_modelfree_actorcritic.py uses and evaluates with the
# existing tester:
#
#   python train_bc_policy.py --min_return 200
#   python test_policy.py --actor_type obs --actor actor_bc.pt --episodes 50
#
# Observations are fed raw, exactly as the environment emits them, because
# test_policy.py passes raw observations at evaluation time.

import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn

from models import ActorObs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
ACTION_NAMES = ["idle", "left", "main", "right"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, model_name, epoch, directory=CHECKPOINT_DIR):
    """Same naming as train_modelfree_actorcritic.py, so the existing tooling reads these."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{model_name}_{timestamp}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), filename)
    return filename


def episode_returns(ep_index, rewards):
    """Total return per episode id, as (ids, returns) aligned by position."""
    ids = np.unique(ep_index)
    order = np.argsort(ep_index, kind="stable")
    sorted_ep = ep_index[order]
    sums = np.add.reduceat(
        rewards[order], np.searchsorted(sorted_ep, ids, side="left")
    )
    return ids, sums


def load_demos(path, min_return, n_episodes, val_frac, seed):
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    ep_index = data["ep_index"]
    rewards = data["rewards"].astype(np.float32)

    ids, rets = episode_returns(ep_index, rewards)
    good = ids[rets >= min_return]
    if good.size == 0:
        raise SystemExit(f"No episodes with return >= {min_return} in {path}")

    # Subsample whole episodes so that smaller budgets are nested subsets.
    rng = np.random.default_rng(seed)
    good = good[rng.permutation(good.size)]
    if n_episodes is not None and n_episodes < good.size:
        good = good[:n_episodes]

    # Split by episode, not by transition: steps within an episode are highly
    # correlated, so a transition-level split would leak across the boundary.
    n_val = max(1, int(round(val_frac * good.size))) if good.size > 1 else 0
    val_ids, train_ids = good[:n_val], good[n_val:]
    if train_ids.size == 0:
        raise SystemExit("No training episodes left after the validation split")

    def take(ep_ids):
        m = np.isin(ep_index, ep_ids)
        return (torch.from_numpy(obs[m]), torch.from_numpy(actions[m]))

    tr = take(train_ids)
    va = take(val_ids) if n_val else (None, None)
    stats = dict(
        n_good=int(good.size), n_train_ep=int(train_ids.size), n_val_ep=int(n_val),
        ret_mean=float(rets[np.isin(ids, good)].mean()),
        total_ep=int(ids.size),
    )
    return tr, va, stats


@torch.no_grad()
def evaluate(actor, obs, actions, batch_size):
    actor.eval()
    ce_sum, correct, n = 0.0, 0, len(obs)
    per_action_hit = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    per_action_tot = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    for i in range(0, n, batch_size):
        o = obs[i:i + batch_size].to(DEVICE)
        a = actions[i:i + batch_size].to(DEVICE)
        dist = actor(o)
        ce_sum += float(-dist.log_prob(a).sum())
        pred = dist.logits.argmax(-1)
        correct += int((pred == a).sum())
        for k in range(len(ACTION_NAMES)):
            m = a == k
            per_action_tot[k] += int(m.sum())
            per_action_hit[k] += int((pred[m] == k).sum())
    recall = np.divide(per_action_hit, np.maximum(per_action_tot, 1), dtype=np.float64)
    return ce_sum / n, correct / n, recall, per_action_tot


def main():
    p = argparse.ArgumentParser(description="Behaviour cloning on successful human landings")
    p.add_argument("--dataset", default="lunarlander_train_dataset.npz")
    p.add_argument("--min_return", type=float, default=200.0,
                   help="Keep episodes with total return >= this (200 = successful landing)")
    p.add_argument("--n_episodes", type=int, default=None,
                   help="Use only this many demonstration episodes (for data-budget sweeps)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    # Must match world_model.capacity.mlp_hidden_dim, since that is what
    # test_policy.py uses to rebuild ActorObs before loading this checkpoint.
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, "actor_bc.pt"))
    p.add_argument("--save_final", default=None,
                   help="Also write the last-epoch weights here. Validation cross-entropy "
                        "is a poor proxy for closed-loop return, so the two are worth comparing.")
    p.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    p.add_argument("--checkpoint_freq", type=int, default=5,
                   help="Snapshot every N epochs so return can be swept over training, "
                        "the way the world-model runs are evaluated. 0 disables.")
    p.add_argument("--checkpoint_name", default="actor_bc")
    args = p.parse_args()

    set_seed(args.seed)

    for path in (args.out, args.save_final):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    (tr_obs, tr_act), val, stats = load_demos(
        args.dataset, args.min_return, args.n_episodes, args.val_frac, args.seed
    )
    obs_dim = tr_obs.shape[1]

    print(f"Dataset: {args.dataset}")
    print(f"  {stats['total_ep']} episodes total, {stats['n_good']} with return >= "
          f"{args.min_return:.0f} (mean return {stats['ret_mean']:+.1f})")
    print(f"  train {stats['n_train_ep']} episodes / {len(tr_obs)} transitions; "
          f"val {stats['n_val_ep']} episodes / {len(val[0]) if val[0] is not None else 0} transitions")
    counts = np.bincount(tr_act.numpy(), minlength=args.action_dim)
    print("  action mix: " + "  ".join(
        f"{ACTION_NAMES[i]}={counts[i] / counts.sum():.1%}" for i in range(args.action_dim)))
    print(f"Device: {DEVICE}   obs_dim={obs_dim}  hidden_dim={args.hidden_dim}\n")

    actor = ActorObs(obs_dim, args.action_dim, hidden_dim=args.hidden_dim).to(DEVICE)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = len(tr_obs)
    best_val, best_epoch = float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        actor.train()
        perm = torch.randperm(n)
        run_ce, seen = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            o = tr_obs[idx].to(DEVICE)
            a = tr_act[idx].to(DEVICE)
            # Cross-entropy: for a Categorical this is exactly -log p(a|s).
            loss = -actor(o).log_prob(a).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            run_ce += loss.item() * len(idx)
            seen += len(idx)
        sched.step()

        line = f"epoch {epoch:3d}/{args.epochs}  train_ce={run_ce / seen:.4f}"
        if val[0] is not None:
            v_ce, v_acc, recall, tot = evaluate(actor, val[0], val[1], args.batch_size)
            line += f"  val_ce={v_ce:.4f}  val_acc={v_acc:.3f}"
            if v_ce < best_val:
                best_val, best_epoch = v_ce, epoch
                torch.save(actor.state_dict(), args.out)
                line += "  *"
        elif epoch == args.epochs:
            torch.save(actor.state_dict(), args.out)

        if args.checkpoint_freq > 0 and (
            epoch % args.checkpoint_freq == 0 or epoch == args.epochs
        ):
            path = save_checkpoint(actor, args.checkpoint_name, epoch, args.checkpoint_dir)
            line += f"  -> {os.path.basename(path)}"
        print(line)

    if args.save_final:
        torch.save(actor.state_dict(), args.save_final)

    if val[0] is not None:
        _, v_acc, recall, tot = evaluate(actor, val[0], val[1], args.batch_size)
        print("\nfinal-epoch per-action recall on held-out episodes:")
        for i in range(args.action_dim):
            print(f"  {ACTION_NAMES[i]:5s} recall={recall[i]:.3f}  ({tot[i]} samples)")
        print(f"\nbest val_ce={best_val:.4f} at epoch {best_epoch}; saved that checkpoint")

    print(f"\nSaved actor -> {os.path.abspath(args.out)}")
    if args.checkpoint_freq > 0:
        print(f"Per-epoch snapshots -> {args.checkpoint_dir}/{args.checkpoint_name}_*_epoch_*.pt")
    print("Evaluate with (--stochastic matters: argmax breaks the human thruster duty cycle):")
    print(f"  python test_policy.py --actor_type obs --actor {args.out} --episodes 50 --stochastic")


if __name__ == "__main__":
    main()
