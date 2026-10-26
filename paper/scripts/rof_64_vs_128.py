"""Estimator noise of ROF: the same checkpoints with 64 vs 128 Jacobian states.

Reproduces the "Estimator noise" paragraph of the ROF reproducibility appendix.

Inputs (logs/paper_sources/metrics/):
  metrics_seed12345{continuous,resume}_seed12345.txt     64 states, metric seed 12345
  metrics128_seed12345{continuous,resume}_seed{12345,1,2}.txt   128 states, seeds 12345/1/2

Run tags: continuous = 12345-5090-C, resume = 12345-5090-A.

Usage:
  python paper/scripts/rof_64_vs_128.py [--dir logs/paper_sources/metrics] [--field jac_rof]
                                        [--window_start 400] [--window_end 500]
  (from the repository root)
"""
import argparse
import math
import os
import re

import numpy as np

ARMS = {"continuous": "12345-5090-C", "resume": "12345-5090-A"}
SEEDS = (12345, 1, 2)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("logs", "paper_sources", "metrics"))
    ap.add_argument("--field", default="jac_rof")
    ap.add_argument("--window_start", type=int, default=400)
    ap.add_argument("--window_end", type=int, default=500)
    args = ap.parse_args()
    f = args.field

    for arm, tag in ARMS.items():
        m64 = parse_metrics(os.path.join(args.dir, f"metrics_seed12345{arm}_seed12345.txt"))
        m128 = {s: parse_metrics(os.path.join(args.dir, f"metrics128_seed12345{arm}_seed{s}.txt"))
                for s in SEEDS}
        epochs = sorted(e for e in m64 if f in m64[e] and all(f in m128[s].get(e, {}) for s in SEEDS))

        a = np.array([m64[e][f] for e in epochs])
        b = np.array([m128[12345][e][f] for e in epochs])
        diff = a - b
        r = np.corrcoef(a, b)[0, 1]

        print(f"=== {tag} ({arm}), field {f}, {len(epochs)} checkpoints, epochs {epochs[0]}-{epochs[-1]}")
        print(f"  Same metric seed (12345), 64 vs 128 states:")
        print(f"    mean 64 = {a.mean():.4f}   mean 128 = {b.mean():.4f}")
        print(f"    per-checkpoint Pearson r = {r:.2f}")
        print(f"    mean |64 - 128| = {np.abs(diff).mean():.4f}")
        print(f"    within-run s.d. (128 states, seed 12345) = {b.std(ddof=1):.4f}")

        win = [e for e in epochs if args.window_start <= e <= args.window_end]
        n = len(win) * len(SEEDS)
        per_seed = np.array([[m128[s][e][f] for s in SEEDS] for e in win])
        sd_seed = per_seed.std(axis=1, ddof=1).mean()
        sd_pair = diff.std(ddof=1) / math.sqrt(2)
        print(f"  Standard error of a window mean, epochs {win[0]}-{win[-1]} "
              f"({len(win)} checkpoints x {len(SEEDS)} metric seeds = {n} values):")
        print(f"    from s.d. across metric seeds ({sd_seed:.4f}): {sd_seed / math.sqrt(n):.4f}")
        print(f"    from the 64-vs-128 differences ({sd_pair:.4f} per estimate): {sd_pair / math.sqrt(n):.4f}")
        print()


if __name__ == "__main__":
    main()
