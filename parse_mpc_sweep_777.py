"""Parse mpc_eval_logs_seed777.txt in-progress and summarise + plot.

Follows J's protocol: MA-7 CENTRED smoothing over per-checkpoint mean_return.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HEADER_RE = re.compile(r"\[\s*(\d+)\s*/\s*100\]\s+Epoch\s+(\d+)")
MEAN_RE = re.compile(r"^\[Run\]\s+mean_return=([+-]?\d+\.\d+)")
WORST_RE = re.compile(r"worst_return=([+-]?\d+\.\d+)")
EVAL_TIME_RE = re.compile(r"^\[Eval time:\s+([\d.]+)s\]")
VERDICT_RE = re.compile(r"^\[Run\]\s+verdict=(PASS|FAIL)\s+\((\d+)/(\d+)")


def parse_log(path: Path):
    """Return list of dicts with per-checkpoint stats."""
    rows = []
    cur = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = HEADER_RE.search(line)
        if m:
            if cur is not None:
                rows.append(cur)
            cur = {
                "idx": int(m.group(1)),
                "epoch": int(m.group(2)),
                "mean_return": None,
                "worst_return": None,
                "eval_time_s": None,
                "verdict": None,
                "checks_passed": None,
            }
            continue
        if cur is None:
            continue
        m = MEAN_RE.search(line)
        if m:
            cur["mean_return"] = float(m.group(1))
            m2 = WORST_RE.search(line)
            if m2:
                cur["worst_return"] = float(m2.group(1))
            continue
        m = VERDICT_RE.search(line)
        if m:
            cur["verdict"] = m.group(1)
            cur["checks_passed"] = int(m.group(2))
            continue
        m = EVAL_TIME_RE.search(line)
        if m:
            cur["eval_time_s"] = float(m.group(1))
            continue
    if cur is not None:
        rows.append(cur)
    return rows


def centred_ma(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Centred moving average with edge shrinkage (same length as x)."""
    n = len(x)
    half = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(x[lo:hi])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="mpc_eval_logs_seed777.txt")
    p.add_argument("--out_plot", default="mpc_sweep_777_progress.png")
    args = p.parse_args()

    rows = parse_log(Path(args.log))
    complete = [r for r in rows if r["mean_return"] is not None]
    print(f"Parsed {len(rows)} checkpoint headers; {len(complete)} with mean_return recorded.")

    if not complete:
        print("No complete checkpoints yet.")
        return

    epochs = np.array([r["epoch"] for r in complete])
    means = np.array([r["mean_return"] for r in complete])
    worsts = np.array([r["worst_return"] for r in complete])
    times = np.array([r["eval_time_s"] for r in complete if r["eval_time_s"] is not None])
    verdicts = [r["verdict"] for r in complete]
    passes = [r["checks_passed"] for r in complete if r["checks_passed"] is not None]

    ma7 = centred_ma(means, 7)

    argmax_raw = int(np.argmax(means))
    argmax_ma = int(np.nanargmax(ma7))
    argmin_raw = int(np.argmin(means))

    # Spearman vs epoch (monotonicity gauge)
    from scipy.stats import spearmanr
    rho_epoch = spearmanr(epochs, means).correlation
    rho_epoch_smoothed = spearmanr(epochs, ma7).correlation

    print()
    print(f"{'='*72}")
    print(f"MPC seed-777 sweep — progress summary  ({len(complete)}/100 checkpoints)")
    print(f"{'='*72}")
    print(f"Epoch range     : {epochs.min()} .. {epochs.max()}")
    print(f"Raw mean_return : mean {means.mean():+7.2f}   median {np.median(means):+7.2f}"
          f"   min {means.min():+7.2f}   max {means.max():+7.2f}")
    print(f"Worst_return    : mean {worsts.mean():+7.2f}   min  {worsts.min():+7.2f}")
    print()
    print(f"Raw argmax      : epoch {epochs[argmax_raw]:>3d}   value {means[argmax_raw]:+7.2f}")
    print(f"MA-7 argmax     : epoch {epochs[argmax_ma]:>3d}   value {ma7[argmax_ma]:+7.2f}")
    print(f"Raw argmin      : epoch {epochs[argmin_raw]:>3d}   value {means[argmin_raw]:+7.2f}")
    print()
    print(f"Spearman(mean_return vs epoch)          = {rho_epoch:+.3f}   (J's 777 run: +0.86)")
    print(f"Spearman(MA-7 smoothed  vs epoch)       = {rho_epoch_smoothed:+.3f}")
    print()

    if times.size:
        avg_t = float(np.mean(times))
        remaining = 100 - len(complete)
        print(f"Avg eval time   : {avg_t:.1f}s / ckpt   ({avg_t/60:.1f} min)")
        print(f"Remaining       : {remaining} ckpts  -> ~{remaining*avg_t/3600:.1f} h")

    print()
    verd_pass = sum(1 for v in verdicts if v == "PASS")
    verd_fail = sum(1 for v in verdicts if v == "FAIL")
    print(f"Verdicts        : PASS {verd_pass}   FAIL {verd_fail}")
    if passes:
        print(f"Checks passed   : mean {np.mean(passes):.2f} / 7   median {int(np.median(passes))}")

    ref = {
        "paper (12345) peak (smoothed)": 153.0,
        "paper (12345) peak (raw)": 166.6,
        "J's 777 peak (smoothed, ep 495)": 199.2,
        "J's 777 start (raw, ep 5)": 54.7,
        "J's 777 final-10 mean": 191.5,
    }
    print()
    print("Reference numbers:")
    for k, v in ref.items():
        print(f"  {k:<40s} {v:+7.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.axhline(153.0, color="tab:orange", ls="--", lw=1.2, alpha=0.7,
               label="paper (seed 12345) peak smoothed +153")
    ax.axhline(199.2, color="tab:green", ls="--", lw=1.2, alpha=0.7,
               label="J's seed 777 peak smoothed +199")

    ax.plot(epochs, means, "o-", color="0.55", ms=4, lw=1.0, alpha=0.85,
            label="raw per-checkpoint mean (20 eps)")
    ax.plot(epochs, ma7, "-", color="tab:blue", lw=2.4,
            label="MA-7 centred (matches paper/J protocol)")

    ax.plot(epochs[argmax_ma], ma7[argmax_ma], "*", color="tab:red", ms=16,
            zorder=5, label=f"MA-7 argmax: ep {epochs[argmax_ma]} ({ma7[argmax_ma]:+.1f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MPC mean return (20 episodes / checkpoint)")
    ax.set_title(f"MPC seed-777 sweep — in progress ({len(complete)}/100 checkpoints)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=140)
    print(f"\nSaved plot -> {args.out_plot}")



if __name__ == "__main__":
    main()
