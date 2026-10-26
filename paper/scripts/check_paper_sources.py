"""Recompute paper numbers from the logs in logs/ and logs/paper_sources/.

  1. Within-run Spearman of pool-averaged ROF vs MA-7 MPC (tab:sign rows of the RTX 5090 runs)
  2. Late validation loss and good-pool ROF of 12345-5090-C and 12345-5090-Z (epochs >= 400)
  3. MF-AC real-environment transitions up to the epoch-760 checkpoint and to epoch 1000
  4. A2C across world-model checkpoints (tab:ac-summary)
  5. CEM-MPC speed on the RTX 5090

Usage:
  python paper/scripts/check_paper_sources.py   (from the repository root)
"""
import glob
import math
import os
import re

LOGS = "logs"
SRC = os.path.join(LOGS, "paper_sources")
DATASET_STEPS = 180_916


def parse_metrics(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"=== Epoch (\d+) ===", line.strip())
        if m:
            cur = int(m.group(1)); out[cur] = {}
            continue
        if cur is not None:
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)", line):
                try:
                    out[cur][k] = float(v)
                except ValueError:
                    pass
    return out


def parse_mpc(path):
    returns, cur = {}, None
    for line in open(path):
        m = re.search(r"\[\d+/\d+\] Epoch (\d+) --", line)
        if m:
            cur = int(m.group(1)); returns[cur] = []
            continue
        m = re.search(r"\[Episode\s+\d+\]\s+return=\s*([-+0-9.]+)", line)
        if m and cur is not None:
            returns[cur].append(float(m.group(1)))
    return returns


def ma(xs, w=7):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def rank(x):
    idx = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((a - my) ** 2 for a in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(rank(x), rank(y))


def seed_mean(runs, epochs, field):
    return [sum(m[e][field] for m in runs) / len(runs) for e in epochs]


def sign_correlations():
    print("=== 1. tab:sign: Spearman(pool-averaged ROF, MA-7 MPC)")
    cases = [
        ("12345-5090-A", "mpc_eval_logs_12345-5090-A.txt", "metrics128_seed12345resume_seed*.txt"),
        ("12345-5090-C", "mpc_eval_logs_12345-5090-C.txt", "metrics128_seed12345continuous_seed*.txt"),
        ("12345-5090-Z", "mpc_eval_logs_12345-5090-Z.txt", "metrics128_seed12345zonly_seed*.txt"),
        ("777-5090-A", "mpc_eval_logs_777-5090-A.txt", "metrics_seed777resume_seed12345.txt"),
    ]
    for tag, mpc_file, pattern in cases:
        mpc = parse_mpc(os.path.join(LOGS, mpc_file))
        files = sorted(glob.glob(os.path.join(SRC, "metrics", pattern)))
        runs = [parse_metrics(f) for f in files]
        epochs = sorted(e for e in mpc if len(mpc[e]) == 20
                        and all("jac_rof" in m.get(e, {}) for m in runs))
        sm = ma([sum(mpc[e]) / len(mpc[e]) for e in epochs])
        rof = [(a + b) / 2 for a, b in zip(seed_mean(runs, epochs, "jac_rof"),
                                           seed_mean(runs, epochs, "jac_rof_bad"))]
        print(f"  {tag:13} rho_s = {spearman(rof, sm):+.2f}   "
              f"({len(files)} metric seed(s), epochs {epochs[0]}-{epochs[-1]}, n={len(epochs)})")


def late_loss_and_rof():
    print("\n=== 2. Late validation loss and good-pool ROF, epochs >= 400")
    for tag, arm in (("12345-5090-C", "continuous"), ("12345-5090-Z", "zonly")):
        vals = {}
        for line in open(os.path.join(SRC, "training", f"train_worldmodel_logs_{tag}.txt")):
            m = re.search(r"\[WorldModel\] Epoch (\d+), .*val_loss=([-+0-9.]+)", line)
            if m:
                vals[int(m.group(1))] = float(m.group(2))
        late = [v for e, v in vals.items() if e >= 400]
        runs = [parse_metrics(f) for f in
                sorted(glob.glob(os.path.join(SRC, "metrics", f"metrics128_seed12345{arm}_seed*.txt")))]
        epochs = sorted(e for e in runs[0] if e >= 400 and all("jac_rof" in m.get(e, {}) for m in runs))
        rof = seed_mean(runs, epochs, "jac_rof")
        print(f"  {tag:13} val_loss = {sum(late) / len(late):.2f} ({len(late)} epochs)   "
              f"good-pool ROF = {sum(rof) / len(rof):.3f} ({len(epochs)} checkpoints x {len(runs)} seeds)")


def mfac_transitions():
    print("\n=== 3. MF-AC transitions (log holds a first run to epoch 616 and a resume from epoch 611)")
    segments, cur = [], None
    for line in open(os.path.join(SRC, "training", "train_modelfree_actorcritic_logs.txt")):
        if line.startswith("A2C training"):
            cur = {}; segments.append(cur)
            continue
        m = re.match(r"\[Epoch (\d+)\] transitions=(\d+)", line)
        if m and cur is not None:
            cur[int(m.group(1))] = int(m.group(2))
    first, resumed = segments
    start = min(resumed)
    path = {e: n for e, n in first.items() if e < start}
    path.update(resumed)
    for upto in (760, 1000):
        total = sum(n for e, n in path.items() if e <= upto)
        print(f"  to epoch {upto:4}: {total:,} transitions = {total / DATASET_STEPS:.2f}x the "
              f"{DATASET_STEPS:,}-step dataset")
    naive = sum(list(first.values()) + list(resumed.values()))
    print(f"  (summing every logged epoch line instead: {naive:,})")


def parse_ac_sweep(path):
    ckpts, cur = [], None
    for line in open(path):
        m = re.search(r"\[\d+/\d+\] Actor epoch (\d+) --", line)
        if m:
            cur = (int(m.group(1)), []); ckpts.append(cur)
            continue
        m = re.match(r"\[Episode\s+\d+\]\s+return=\s*([-+0-9.]+)", line)
        if m and cur is not None:
            cur[1].append(float(m.group(1)))
    return ckpts


def ac_table():
    print("\n=== 4. tab:ac-summary (100 episodes per A2C checkpoint)")
    print(f"  {'WM':>4} | {'BBM ep':>6} {'mean':>7} {'worst':>7} {'cat':>3} {'perf':>4} | "
          f"{'SBW ep':>6} {'mean':>7} {'worst':>7} {'cat':>3} {'perf':>4} | #safe")
    for wm in (50, 60, 100, 200, 250, 280, 300, 310, 315):
        name = f"rl_eval_logs_wm{wm}AC_100ep.txt"
        path = os.path.join(SRC, "a2c_sweeps", name)
        if not os.path.exists(path):
            path = os.path.join(LOGS, name)
        rows = []
        for ep, rets in parse_ac_sweep(path):
            rows.append(dict(ep=ep, mean=sum(rets) / len(rets), worst=min(rets),
                             cat=sum(r < -100 for r in rets), perf=sum(r > 200 for r in rets)))
        bbm = max(rows, key=lambda r: r["mean"])
        sbw = max(rows, key=lambda r: r["worst"])
        safe = sum(r["cat"] == 0 for r in rows)
        fmt = lambda r: f"{r['ep']:>6} {r['mean']:>+7.1f} {r['worst']:>+7.1f} {r['cat']:>3} {r['perf']:>4}"
        print(f"  {wm:>4} | {fmt(bbm)} | {fmt(sbw)} | {safe}/{len(rows)}")


def mpc_speed():
    print("\n=== 5. CEM-MPC speed on the RTX 5090 (held-out runs, environment steps per second)")
    for path in sorted(glob.glob(os.path.join(LOGS, "eval_heldout", "heldout_*_mpc_*.txt"))):
        steps = secs = 0.0
        for line in open(path):
            m = re.search(r"\[Episode\s+\d+\].*steps=(\d+).*elapsed=([0-9.]+)s", line)
            if m:
                steps += int(m.group(1)); secs += float(m.group(2))
        print(f"  {os.path.basename(path):65} {steps / secs:5.1f} steps/s")


if __name__ == "__main__":
    sign_correlations()
    late_loss_and_rof()
    mfac_transitions()
    ac_table()
    mpc_speed()
