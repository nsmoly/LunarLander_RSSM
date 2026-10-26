"""Terminal-event reward fidelity of world-model checkpoints.

Question: does the world model still know that a crash is bad, and see it coming
from a planning horizon away?

Every reward metric in eval_metrics.py is an RMSE averaged over all transitions,
so the one terminal step per episode (about -100 for a crash, +100 for a
landing) is diluted by the flight steps around it, and the sign is discarded.
This script measures the model on the terminal transitions alone, signed.

For each logged episode that ended in a real terminal event, and for each
horizon H:
  1. posterior warmup on W real steps, ending H steps before the terminal step
  2. open-loop rollout through the prior (means, no observations) using the
     logged actions, up to and including the terminal transition
  3. compare predicted reward / done probability with the logged values

H = 0 is the teacher-forced reference: the posterior is fed every real
observation through the terminal step, exactly as in training, so the reward
head sees the post-crash observation. H >= 1 uses the prior for the last H
steps, which is what the planner sees; even H = 1 requires the model to
anticipate the terminal event rather than observe it.

Bulk control: the same rollout on fixed random non-terminal windows that end at
least max(H) steps before their episode's end, so that "tail gets worse while
bulk improves" is visible.

No sampling anywhere (posterior and prior means, fixed windows), so each
checkpoint gives one deterministic row set and no seed averaging is needed.

Output: one tab-separated row per (epoch, horizon, category), appended per
checkpoint; epochs already present in the output are skipped on re-run.

Usage:
    python tools/eval_terminal_fidelity.py --checkpoints_dir checkpoints_seed777 \
        --output logs/paper_sources/terminal_fidelity/seed777.tsv
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import WorldModel  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRASH_R = -50.0
LAND_R = 50.0

COLUMNS = [
    "epoch", "horizon", "category", "n",
    "pred_T", "true_T", "err_T", "abs_err_T", "hit_rate",
    "extreme3", "hit3",
    "done_T", "done_max3",
    "cum_pred", "cum_true", "cum_err",
]


def load_world_model(config_path, ckpt_path, obs_dim):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = (yaml.safe_load(f) or {}).get("world_model", {})
    cap = cfg.get("capacity", {})
    hidden_dim = int(cap.get("hidden_dim", 256))
    wm = WorldModel(
        obs_dim=obs_dim,
        action_dim=int(cfg.get("action_dim", 4)),
        latent_dim=int(cap.get("latent_dim", 16)),
        hidden_dim=hidden_dim,
        gru_num_layers=int(cap.get("gru_num_layers", 1)),
        mlp_hidden_dim=int(cap.get("mlp_hidden_dim", hidden_dim)),
    ).to(DEVICE)
    payload = torch.load(ckpt_path, map_location=DEVICE)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    wm.load_state_dict(sd)
    wm.eval()
    return wm


def load_episodes(path):
    d = np.load(path)
    order = np.lexsort((d["step_index"], d["ep_index"]))
    ep = d["ep_index"][order]
    cols = {
        "obs": d["obs"][order].astype(np.float32),
        "next_obs": d["next_obs"][order].astype(np.float32),
        "act": d["actions"][order].astype(np.int64),
        "rew": d["rewards"][order].astype(np.float32),
        "done": d["dones"][order].astype(np.int64),
    }
    cuts = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(ep)]])
    return [{k: v[s:e] for k, v in cols.items()} for s, e in zip(starts, ends)]


def build_windows(episodes, warmup, max_h, n_bulk, seed):
    """Return {category: [(episode_idx, T)]} where T is the index of the last
    transition scored. Every window needs warmup + max_h transitions before and
    including T, so all horizons score the same events."""
    need = warmup + max_h
    windows = {"crash": [], "land": [], "bulk": []}
    bulk_pool = []
    counts = {"episodes": len(episodes), "too_short": 0, "not_terminal": 0}
    for i, e in enumerate(episodes):
        n = len(e["rew"])
        if n < need:
            counts["too_short"] += 1
            continue
        last = n - 1
        r = float(e["rew"][last])
        if e["done"][last] == 1 and r <= CRASH_R:
            windows["crash"].append((i, last))
        elif e["done"][last] == 1 and r >= LAND_R:
            windows["land"].append((i, last))
        else:
            counts["not_terminal"] += 1
        # Bulk windows stay clear of the final max_h steps so they never overlap
        # the terminal approach, and contain no terminal-sized reward.
        for T in range(need - 1, n - max_h):
            if np.abs(e["rew"][T - need + 1:T + 1]).max() < LAND_R:
                bulk_pool.append((i, T))
    rng = np.random.default_rng(seed)
    if bulk_pool:
        pick = rng.choice(len(bulk_pool), size=min(n_bulk, len(bulk_pool)), replace=False)
        windows["bulk"] = [bulk_pool[k] for k in sorted(pick)]
    return windows, counts


def stack_batch(episodes, wins, warmup, horizon):
    """Scored steps are the last max(horizon, warmup) transitions ending at T:
    the H prior steps for H >= 1, or the W posterior steps for H = 0."""
    obs0, w_act, w_nobs, ol_act, gt_rew = [], [], [], [], []
    scored = horizon if horizon > 0 else warmup
    for i, T in wins:
        e = episodes[i]
        k0 = T - horizon - warmup + 1
        obs0.append(e["obs"][k0])
        w_act.append(e["act"][k0:k0 + warmup])
        w_nobs.append(e["next_obs"][k0:k0 + warmup])
        ol_act.append(e["act"][T - horizon + 1:T + 1])
        gt_rew.append(e["rew"][T - scored + 1:T + 1])
    t = lambda x, dt: torch.tensor(np.stack(x), dtype=dt, device=DEVICE)
    return (t(obs0, torch.float32), t(w_act, torch.long), t(w_nobs, torch.float32),
            t(ol_act, torch.long), t(gt_rew, torch.float32))


@torch.no_grad()
def rollout(wm, obs0, w_act, w_nobs, ol_act):
    """Same state/reward alignment as training: predict_reward(h_{t+1}, z_{t+1})
    is the reward of transition (s_t, a_t). Returns predictions from the prior
    steps, or from the posterior warmup steps when there are no prior steps."""
    rssm = wm.rssm
    B = obs0.shape[0]
    A = rssm.action_dim
    h = rssm.init_hidden(B, DEVICE)
    z = torch.zeros(B, rssm.latent_dim, device=DEVICE)
    h = rssm.update_hidden(h, z, torch.zeros(B, A, device=DEVICE))
    z, _ = rssm.posterior(h, obs0)
    post_r, post_d = [], []
    for j in range(w_act.shape[1]):
        h = rssm.update_hidden(h, z, F.one_hot(w_act[:, j], A).float())
        z, _ = rssm.posterior(h, w_nobs[:, j])
        post_r.append(wm.predict_reward(h, z))
        post_d.append(torch.sigmoid(wm.predict_done_logits(h, z)))
    if ol_act.shape[1] == 0:
        return torch.stack(post_r, 1), torch.stack(post_d, 1)
    rews, dones = [], []
    for j in range(ol_act.shape[1]):
        h = rssm.update_hidden(h, z, F.one_hot(ol_act[:, j], A).float())
        z, _ = rssm.prior(h)
        rews.append(wm.predict_reward(h, z))
        dones.append(torch.sigmoid(wm.predict_done_logits(h, z)))
    return torch.stack(rews, 1), torch.stack(dones, 1)


def summarise(category, pred, done_p, gt_rew):
    pT, tT = pred[:, -1], gt_rew[:, -1]
    k = min(3, pred.shape[1])
    if category == "crash":
        hit = (pT <= CRASH_R).float().mean().item()
        ext = pred[:, -k:].min(dim=1).values
        hit3 = (ext <= CRASH_R).float().mean().item()
    elif category == "land":
        hit = (pT >= LAND_R).float().mean().item()
        ext = pred[:, -k:].max(dim=1).values
        hit3 = (ext >= LAND_R).float().mean().item()
    else:
        hit = hit3 = float("nan")
        ext = torch.full_like(pT, float("nan"))
    return {
        "n": pred.shape[0],
        "pred_T": pT.mean().item(),
        "true_T": tT.mean().item(),
        "err_T": (pT - tT).mean().item(),
        "abs_err_T": (pT - tT).abs().mean().item(),
        "hit_rate": hit,
        "extreme3": ext.mean().item(),
        "hit3": hit3,
        "done_T": done_p[:, -1].mean().item(),
        "done_max3": done_p[:, -k:].max(dim=1).values.mean().item(),
        "cum_pred": pred.sum(1).mean().item(),
        "cum_true": gt_rew.sum(1).mean().item(),
        "cum_err": (pred.sum(1) - gt_rew.sum(1)).mean().item(),
    }


def find_checkpoints(ckpt_dir, stride, epoch_min, epoch_max):
    by_epoch = {}
    for p in glob.glob(os.path.join(ckpt_dir, "world_model_*_epoch_*.pt")):
        m = re.search(r"epoch_(\d+)\.pt$", p)
        if not m:
            continue
        ep = int(m.group(1))
        # Duplicate epochs (a restarted run) resolve to the newest file.
        if ep not in by_epoch or os.path.getmtime(p) > os.path.getmtime(by_epoch[ep]):
            by_epoch[ep] = p
    keep = [(ep, p) for ep, p in by_epoch.items()
            if ep % stride == 0 and epoch_min <= ep <= epoch_max]
    return sorted(keep)


def done_epochs(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("epoch"):
                continue
            parts = line.split("\t")
            if parts and parts[0].isdigit():
                out.add(int(parts[0]))
    return out


def evaluate_run(checkpoints_dir, output, dataset="lunarlander_val_dataset.npz",
                 config="config.yaml", horizons=(25, 10, 1, 0), warmup=5,
                 n_bulk=256, bulk_seed=12345, epoch_stride=5,
                 epoch_min=0, epoch_max=10**9, label=None):
    horizons = sorted({int(h) for h in horizons}, reverse=True)
    episodes = load_episodes(dataset)
    windows, counts = build_windows(episodes, warmup, max(horizons), n_bulk, bulk_seed)
    if not windows["crash"] or not windows["land"]:
        raise SystemExit(f"Need both crash and landing events, got "
                         f"{len(windows['crash'])} crashes / {len(windows['land'])} landings")

    ckpts = find_checkpoints(checkpoints_dir, epoch_stride, epoch_min, epoch_max)
    if not ckpts:
        raise SystemExit(f"No world_model_*_epoch_*.pt checkpoints matched in {checkpoints_dir}")
    skip = done_epochs(output)
    todo = [(ep, p) for ep, p in ckpts if ep not in skip]

    summary = (f"run={label or os.path.basename(os.path.normpath(checkpoints_dir))} "
               f"dataset={os.path.basename(dataset)} warmup={warmup} horizons={horizons} "
               f"crash={len(windows['crash'])} land={len(windows['land'])} "
               f"bulk={len(windows['bulk'])} episodes={counts['episodes']} "
               f"too_short={counts['too_short']} not_terminal={counts['not_terminal']}")
    print(f"# {summary}")
    print(f"# checkpoints: {len(ckpts)} matched, {len(ckpts) - len(todo)} already done, "
          f"{len(todo)} to evaluate on {DEVICE}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    new_file = not os.path.exists(output)
    obs_dim = episodes[0]["obs"].shape[1]

    batches = {(h, c): stack_batch(episodes, windows[c], warmup, h)
               for h in horizons for c in ("crash", "land", "bulk") if windows[c]}

    with open(output, "a", encoding="utf-8") as f:
        if new_file:
            f.write(f"# {summary}\n")
            f.write("\t".join(COLUMNS) + "\n")
        for ep, path in todo:
            t0 = time.time()
            wm = load_world_model(config, path, obs_dim)
            rows = []
            for (h, c), (obs0, w_act, w_nobs, ol_act, gt_rew) in batches.items():
                pred, done_p = rollout(wm, obs0, w_act, w_nobs, ol_act)
                s = summarise(c, pred, done_p, gt_rew)
                rows.append({"epoch": ep, "horizon": h, "category": c, **s})
            for r in rows:
                f.write("\t".join(
                    str(r[k]) if k in ("epoch", "horizon", "category", "n") else f"{r[k]:.4f}"
                    for k in COLUMNS) + "\n")
            f.flush()
            crash25 = next(r for r in rows if r["category"] == "crash" and r["horizon"] == horizons[0])
            print(f"epoch {ep:3d}  crash pred_T@H{horizons[0]}={crash25['pred_T']:+8.2f} "
                  f"hit={crash25['hit_rate']:.2f}  [{time.time() - t0:.1f}s]", flush=True)
            del wm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="lunarlander_val_dataset.npz")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--horizons", default="25,10,1,0",
                    help="comma list; 0 = teacher-forced reference, 1 = one prior step")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n_bulk", type=int, default=256)
    ap.add_argument("--bulk_seed", type=int, default=12345)
    ap.add_argument("--epoch_stride", type=int, default=5)
    ap.add_argument("--epoch_min", type=int, default=0)
    ap.add_argument("--epoch_max", type=int, default=10**9)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    evaluate_run(args.checkpoints_dir, args.output, args.dataset, args.config,
                 [int(x) for x in args.horizons.split(",")], args.warmup, args.n_bulk,
                 args.bulk_seed, args.epoch_stride, args.epoch_min, args.epoch_max, args.label)


if __name__ == "__main__":
    main()
