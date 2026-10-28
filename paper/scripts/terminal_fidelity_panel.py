"""Terminal-event fidelity across the healthy and collapsing runs we have on disk.

Runs eval_terminal_fidelity.py on each run (resumable; finished epochs are
skipped), then compares each series with the run's MPC sweep:

  - early (epochs 100-200) vs late (400-500) window means, and the change
  - within-run Spearman between the MA-7 series and MPC MA-7 (epochs > 50)
  - across runs: Spearman of the change against MPC drawdown (n = 8, descriptive)

Write down the prediction before running --analyze_only for the first time.

Usage:
    python paper/scripts/terminal_fidelity_panel.py                  # evaluate all runs, then analyse
    python paper/scripts/terminal_fidelity_panel.py --analyze_only   # analyse existing outputs
    python paper/scripts/terminal_fidelity_panel.py --runs v7,c12345 # a subset
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from eval_terminal_fidelity import evaluate_run  # noqa: E402
from parse_mpc_sweep_777 import centred_ma, parse_log  # noqa: E402

CROF = ROOT.parent / "rof-reproduction"

RUNS = [
    dict(key="v7", label="V7 12345 cont (3080)", outcome="collapse",
         ckpt=CROF / "wm_checkpoints/v7_12345_continuous",
         mpc=CROF / "results/mpc_v7_12345_continuous.txt"),
    dict(key="paper", label="12345 resume@300 (3080)", outcome="collapse",
         ckpt=ROOT / "checkpoints_paper_seed12345resumed",
         mpc=ROOT / "logs/mpc_eval_logs.txt"),
    dict(key="zonly3080", label="z-only 12345 (3080)", outcome="collapse",
         ckpt=CROF / "wm_checkpoints/zonly_12345_3080",
         mpc=CROF / "results/mpc_zonly_12345_3080.txt"),
    dict(key="reset12345", label="12345 reset@300 (5090)", outcome="degraded",
         ckpt=ROOT / "checkpoints_seed12345resume",
         mpc=ROOT / "logs/mpc_eval_logs_12345-5090-A.txt"),
    dict(key="zonly5090", label="z-only 12345 (5090)", outcome="dip+rebound",
         ckpt=ROOT / "checkpoints_seed12345zonly",
         mpc=ROOT / "logs/mpc_eval_logs_12345-5090-Z.txt"),
    dict(key="c12345", label="12345 cont (5090)", outcome="healthy",
         ckpt=ROOT / "checkpoints_seed12345continuous",
         mpc=ROOT / "logs/mpc_eval_logs_12345-5090-C.txt"),
    dict(key="c777", label="777 cont (3080)", outcome="healthy",
         ckpt=ROOT / "checkpoints_seed777",
         mpc=ROOT / "logs/mpc_eval_logs_777-3080-C.txt"),
    dict(key="reset777", label="777 reset@300", outcome="healthy",
         ckpt=ROOT / "checkpoints_seed777resume",
         mpc=ROOT / "logs/mpc_eval_logs_777-5090-A.txt"),
]

# (category, column) pairs reported at the primary horizon.
SERIES = [
    ("crash", "pred_T"),     # predicted reward at the crash step (true is -100)
    ("crash", "extreme3"),   # most negative predicted reward over the last 3 steps
    ("crash", "hit3"),       # fraction of crashes predicted below -50 within the last 3 steps
    ("crash", "done_max3"),  # predicted termination probability near the crash
    ("crash", "cum_err"),    # signed error of the H-step return into a crash
    ("land", "extreme3"),    # highest predicted reward over the last 3 steps (true +100)
    ("land", "hit3"),
    ("land", "cum_err"),     # signed: positive = imagined landing return exceeds the real one
    ("bulk", "cum_err"),     # control: same on ordinary non-terminal windows
]

EARLY = (100, 200)
LATE = (400, 500)
MIN_EPOCH = 50


def read_tsv(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        rows = csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t")
        for r in rows:
            key = (int(r["horizon"]), r["category"])
            data.setdefault(key, {})[int(r["epoch"])] = r
    return data


def series(data, horizon, category, column):
    d = data.get((horizon, category), {})
    eps = np.array(sorted(d))
    vals = np.array([float(d[e][column]) for e in eps], dtype=float)
    return eps, vals


def load_mpc(path):
    rows = [r for r in parse_log(Path(path)) if r["mean_return"] is not None]
    ep = np.array([r["epoch"] for r in rows])
    y = np.array([r["mean_return"] for r in rows], dtype=float)
    return ep, y, centred_ma(y, 7)


def window_mean(eps, vals, lo, hi):
    m = (eps >= lo) & (eps <= hi)
    return float(np.nanmean(vals[m])) if m.any() else float("nan")


def mpc_outcome(ep, y, ma):
    post = ep > MIN_EPOCH
    peak = float(np.nanmax(ma[post]))
    final10 = float(np.mean(y[-10:]))
    return peak, final10, peak - final10


def within_rho(eps, vals_ma, mpc_ep, mpc_ma):
    idx = {int(e): i for i, e in enumerate(mpc_ep)}
    shared = [(v, mpc_ma[idx[int(e)]]) for e, v in zip(eps, vals_ma)
              if int(e) > MIN_EPOCH and int(e) in idx and np.isfinite(v)]
    if len(shared) < 5:
        return float("nan")
    a, b = zip(*shared)
    return float(spearmanr(a, b).correlation)


def analyse(runs, out_dir, horizon, plot):
    loaded = []
    for run in runs:
        tsv = out_dir / f"{run['key']}.tsv"
        if not tsv.exists():
            print(f"  [skip] {run['key']}: {tsv} not found")
            continue
        data = read_tsv(tsv)
        mpc_ep, mpc_y, mpc_ma = load_mpc(run["mpc"])
        peak, final10, drawdown = mpc_outcome(mpc_ep, mpc_y, mpc_ma)
        loaded.append(dict(run=run, data=data, mpc_ep=mpc_ep, mpc_ma=mpc_ma,
                           peak=peak, final10=final10, drawdown=drawdown))
    if not loaded:
        raise SystemExit("Nothing to analyse.")

    print(f"\nMPC reference (epochs > {MIN_EPOCH}): MA-7 peak, final-10 raw mean, drawdown")
    for L in loaded:
        print(f"  {L['run']['label']:26s} {L['run']['outcome']:12s} "
              f"peak {L['peak']:+7.1f}  final-10 {L['final10']:+7.1f}  drawdown {L['drawdown']:6.1f}")

    across = {}
    for cat, col in SERIES:
        name = f"{cat}.{col}"
        print(f"\n=== {name}  (H={horizon}, MA-7 over checkpoints) ===")
        print(f"  {'run':26s} {'outcome':12s} {'early':>9s} {'late':>9s} {'late-early':>11s} {'rho(MPC)':>9s}")
        deltas, lates, draws, finals = [], [], [], []
        for L in loaded:
            eps, vals = series(L["data"], horizon, cat, col)
            if len(eps) == 0:
                continue
            ma = centred_ma(vals, 7)
            e, l = window_mean(eps, ma, *EARLY), window_mean(eps, ma, *LATE)
            rho = within_rho(eps, ma, L["mpc_ep"], L["mpc_ma"])
            print(f"  {L['run']['label']:26s} {L['run']['outcome']:12s} "
                  f"{e:+9.3f} {l:+9.3f} {l - e:+11.3f} {rho:+9.2f}")
            deltas.append(l - e); lates.append(l)
            draws.append(L["drawdown"]); finals.append(L["final10"])
        if len(deltas) >= 4 and np.all(np.isfinite(deltas)):
            r1 = spearmanr(deltas, draws).correlation
            r2 = spearmanr(lates, finals).correlation
            across[name] = (r1, r2, len(deltas))
            print(f"  across runs (n={len(deltas)}): rho(late-early, MPC drawdown) = {r1:+.2f}   "
                  f"rho(late, MPC final-10) = {r2:+.2f}")

    if across:
        print(f"\nSummary across runs (n = {len(loaded)}; |rho| > 0.74 needed for p < 0.05 at n = 8)")
        for name, (r1, r2, n) in across.items():
            print(f"  {name:18s} change vs drawdown {r1:+.2f}   late vs final-10 {r2:+.2f}")

    if plot:
        make_plot(loaded, horizon, out_dir / "terminal_fidelity_panel.png")


def make_plot(loaded, horizon, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(loaded)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, L in zip(axes.flat, loaded):
        ax.plot(L["mpc_ep"], L["mpc_ma"], color="black", lw=1.8, label="MPC MA-7")
        ax.set_ylabel("MPC return", fontsize=8)
        ax.axhline(0, color="0.8", lw=0.7)
        ax2 = ax.twinx()
        for (h, cat, style, color, lab) in [
            (horizon, "crash", "-", "#D62728", f"crash extreme3 H{horizon}"),
            (0, "crash", ":", "#D62728", "crash extreme3 H0 (teacher-forced)"),
            (horizon, "land", "-", "#2CA02C", f"land extreme3 H{horizon}"),
            (0, "land", ":", "#2CA02C", "land extreme3 H0 (teacher-forced)"),
        ]:
            eps, vals = series(L["data"], h, cat, "extreme3")
            if len(eps):
                ax2.plot(eps, centred_ma(vals, 7), style, color=color, lw=1.3, label=lab)
        ax2.axhline(-100, color="#D62728", lw=0.5, alpha=0.4)
        ax2.axhline(100, color="#2CA02C", lw=0.5, alpha=0.4)
        ax2.set_ylabel("predicted terminal reward", fontsize=8)
        ax.set_title(f"{L['run']['label']}  [{L['run']['outcome']}]", fontsize=9)
        ax.set_xlim(0, 505)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.legend(*_all_handles(fig), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06 if rows > 1 else 0.10, 1, 1))
    fig.savefig(out_png, dpi=140)
    print(f"\nSaved {out_png}")


def _all_handles(fig):
    seen, handles, labels = set(), [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    return handles, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="", help="comma list of run keys (default: all)")
    ap.add_argument("--out_dir", default="logs/paper_sources/terminal_fidelity")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--epoch_stride", type=int, default=5)
    ap.add_argument("--dataset", default="lunarlander_val_dataset.npz")
    ap.add_argument("--horizon", type=int, default=25, help="primary horizon for the analysis")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    runs = RUNS
    if args.runs:
        want = set(args.runs.split(","))
        unknown = want - {r["key"] for r in RUNS}
        if unknown:
            raise SystemExit(f"Unknown run keys: {sorted(unknown)}; choose from {[r['key'] for r in RUNS]}")
        runs = [r for r in RUNS if r["key"] in want]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    os.chdir(ROOT)

    if not args.analyze_only:
        for run in runs:
            print(f"\n##### {run['key']}: {run['label']} [{run['outcome']}]")
            evaluate_run(str(run["ckpt"]), str(out_dir / f"{run['key']}.tsv"),
                         dataset=args.dataset, epoch_stride=args.epoch_stride, label=run["key"])

    analyse(runs, out_dir, args.horizon, plot=not args.no_plot)


if __name__ == "__main__":
    main()
