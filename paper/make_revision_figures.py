"""Figures added in the revision, written to paper/figures/.

  mpc_all_runs.png   -- MA-7 MPC return for the standard-configuration runs.
  zonly3080_mpc.png  -- MA-7 MPC return for the reward-head ablation.
  zonly_rof.png      -- good-pool ROF over training for the same arms.

two_env_rho.png is no longer in the paper (raw ROF--MPC signs are
coordinate-dependent). The generator remains as fig_two_env().

Run tags follow the paper: <seed>-<GPU>-<protocol>, with protocol C = one
continuous run, R = resumed at epoch 300, A = AdamW moments reset at epoch 300
(weights copied from the C run up to 300), Z = reward head reads z only.

Reads logs from this repository and from the rof-reproduction repository
(https://github.com/jonstraveladventures/rof-reproduction), which is expected
to be cloned next to LunarLander_RSSM.

    python paper/make_revision_figures.py
"""

import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
REPRO = ROOT.parent / "rof-reproduction"
RES = REPRO / "results"
OUT = PAPER / "figures"

sys.path.insert(0, str(ROOT))
from parse_mpc_sweep_777 import centred_ma, parse_log  # noqa: E402

spec = importlib.util.spec_from_file_location("an", REPRO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

WARMUP = 50


# ---------------------------------------------------------------------------
# All standard-configuration runs
# ---------------------------------------------------------------------------
ALL_RUNS = [
    # collapsed (reds)
    ("12345-3080-C", RES / "mpc_v7_12345_continuous.txt", "#8B0000", "-", 2.6, "collapse"),
    ("12345-3080-R", PAPER / "logs" / "mpc_eval_logs.txt", "#E04B2A", "-", 2.6, "collapse"),
    # degraded (amber)
    ("12345-5090-A", ROOT / "mpc_eval_logs_seed12345resume.txt", "#E8A33D", "-", 2.6, "degraded"),
    # healthy (greens)
    ("12345-5090-C", ROOT / "mpc_eval_logs_seed12345continuous.txt", "#1B7F3B", "-", 2.4, "healthy"),
    ("777-3080-C", ROOT / "mpc_eval_logs_seed777.txt", "#4FAE5A", "-", 2.2, "healthy"),
    ("777-5090-A", ROOT / "mpc_eval_logs_seed777resume.txt", "#7FC97F", "--", 2.2, "healthy"),
    ("777-L40S-C", RES / "mpc_dq_f1rs.txt", "#2E8B8B", ":", 2.2, "healthy"),
]


def load_mpc(path):
    rows = [r for r in parse_log(Path(path)) if r["mean_return"] is not None]
    ep = np.array([r["epoch"] for r in rows], dtype=float)
    return ep, centred_ma(np.array([r["mean_return"] for r in rows], dtype=float), 7)


def fig_all_runs():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), height_ratios=[2.2, 1], sharex=True)
    ax.axhline(0, color="0.75", lw=0.9, zorder=0)
    ax.axvline(300, color="0.55", lw=1.1, ls="--", zorder=0)
    ax.text(303, -95, "restart / reset at 300", color="0.4", fontsize=8.5, rotation=90, va="bottom")
    for tag, path, colour, ls, lw, outcome in ALL_RUNS:
        ep, ma = load_mpc(path)
        ax.plot(ep, ma, ls, color=colour, lw=lw, label=f"{tag} ({outcome})",
                zorder=4 if outcome != "healthy" else 3)
        i300 = int(np.argmin(np.abs(ep - 300)))
        ax2.plot(ep[i300:], ma[i300:] - ma[i300], ls, color=colour, lw=lw * 0.85)
    ax.set_ylabel("CEM-MPC mean return (MA-7)")
    ax.set_ylim(-120, 230)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9, ncol=2)
    ax2.axhline(0, color="0.55", lw=1.0)
    ax2.set_xlabel("world-model training epoch")
    ax2.set_ylabel("change since epoch 300")
    ax2.set_xlim(0, 505)
    ax2.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(OUT / "mpc_all_runs.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'mpc_all_runs.png'}")


# ---------------------------------------------------------------------------
# Reward-head ablation
# ---------------------------------------------------------------------------
ZONLY_ARMS = [
    ("12345-3080-Z (z-only head)", RES / "mpc_zonly_12345_3080.txt", "metrics_zonly3080_seed*.txt", "#d62728", 2.6, 5),
    ("12345-3080-C ([h, z] head, control)", RES / "mpc_v7_12345_continuous.txt", "metrics_v7_seed*.txt", "#1f77b4", 2.0, 4),
    ("12345-5090-Z (z-only head)", RES / "mpc_zonly_12345.txt", "metrics_zonly_seed*.txt", "#ff7f0e", 1.6, 3),
    ("777-L40S-C ([h, z] head, healthy reference)", RES / "mpc_dq_f1rs.txt", "metrics_dq_f1rs_seed*.txt", "#2ca02c", 1.6, 2),
]


def zonly_mpc_curve(path):
    mpc = an.parse_mpc(str(path))
    eps = sorted(e for e in mpc if e > WARMUP)
    return eps, an.ma([float(np.mean(mpc[e])) for e in eps])


def zonly_rof_curve(glob_pat):
    runs = [an.parse_metrics(str(f)) for f in sorted(RES.glob(glob_pat))]
    if not runs:
        return [], []
    eps = sorted(e for e in set.intersection(*[set(r) for r in runs]) if e > WARMUP)
    vals = [float(np.mean([r[e]["jac_rof"] for r in runs if "jac_rof" in r.get(e, {})])) for e in eps]
    return eps, an.ma(vals)


def fig_zonly(kind, ylabel, out_name, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    for label, mpc_path, glob_pat, colour, lw, z in ZONLY_ARMS:
        eps, ys = zonly_mpc_curve(mpc_path) if kind == "mpc" else zonly_rof_curve(glob_pat)
        if len(eps) == 0:
            raise SystemExit(f"No data for {label}")
        ax.plot(eps, ys, label=label, color=colour, lw=lw, zorder=z)
    if kind == "mpc":
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("world-model training epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(WARMUP, 505)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / out_name}")


# ---------------------------------------------------------------------------
# Two-environment contrast (same statistics as rof-reproduction's
# make_contrast_figure.py, reimplemented so its repo is not written to)
# ---------------------------------------------------------------------------
PANEL = [101, 202, 303, 404, 505, 606, 707, 808]
C_HEALTHY, C_COLLAPSE, C_777, C_REF = "#0072B2", "#D55E00", "#009E73", "#555555"


def run_stats(mpc_path, metric_glob):
    mpc = an.parse_mpc(str(mpc_path))
    runs = [an.parse_metrics(f) for f in sorted(glob.glob(str(metric_glob)))]
    epochs = sorted(set(mpc) & set.intersection(*[set(m) for m in runs]))
    sm = an.ma([sum(mpc[e]) / len(mpc[e]) for e in epochs])
    comb = [sum(0.5 * m[e]["jac_rof"] + 0.5 * m[e]["jac_rof_bad"] for m in runs) / len(runs) for e in epochs]
    tail = sum(sm[-10:]) / 10
    return {"rho": an.spearman(comb, sm),
            "collapse": (max(sm) - tail) > 0.5 * (max(sm) - min(sm))}


def fig_two_env():
    ll = [(f"seed {s}", run_stats(RES / (f"mpc_sp{s}.txt" if s != 707 else "mpc_sp707_merged.txt"),
                                  RES / f"metrics_sp{s}_seed*.txt"), None) for s in PANEL]
    ll += [("777-L40S-C (ref)", run_stats(RES / "mpc_dq_f1rs.txt", RES / "metrics_dq_f1rs_seed*.txt"), C_777),
           ("12345-3080-R (ref)", run_stats(REPRO / "logs" / "mpc_eval_logs.txt",
                                            REPRO / "logs" / "metrics_eval_logs.txt"), C_REF)]
    rc = [(f"seed {s}", run_stats(RES / f"mpc_r{s}.txt", RES / f"metrics_r{s}_seed*.txt"), None) for s in PANEL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    panels = [("LunarLander (non-Markovian reward): 5/8 collapse", ll),
              ("Reacher (Markovian reward): 0/8 collapse", rc)]
    for ax, (title, entries) in zip(axes, panels):
        entries = sorted(entries, key=lambda t: t[1]["rho"])
        ax.axvline(0, lw=0.8, color="#bbbbbb", zorder=0)
        for y, (name, st, ref_col) in enumerate(entries):
            ref = ref_col is not None
            col = ref_col if ref else (C_COLLAPSE if st["collapse"] else C_HEALTHY)
            ax.plot(st["rho"], y, "D" if ref else "o", ms=8 if ref else 9, color=col, zorder=3)
            right = 0 <= st["rho"] < 0.8
            ax.text(st["rho"] + (0.05 if right else -0.05), y,
                    f"{st['rho']:+.2f}".replace("-", "\N{MINUS SIGN}"),
                    va="center", ha="left" if right else "right", fontsize=8.5, color="#333333")
        ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=9)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("within-run Spearman ρ (pool-averaged ROF vs. MA-7 MPC)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", lw=0.3, color="#eeeeee")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_COLLAPSE, label="collapse"),
               plt.Line2D([], [], marker="o", ls="", color=C_HEALTHY, label="healthy"),
               plt.Line2D([], [], marker="D", ls="", color="#888888", label="reference runs")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "two_env_rho.png", dpi=170)
    plt.close(fig)
    print(f"wrote {OUT / 'two_env_rho.png'}")
    for name, st, _ in ll[-2:]:
        print(f"  {name}: rho = {st['rho']:+.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_all_runs()
    fig_zonly("mpc", "CEM-MPC mean return (MA-7, 20 episodes/checkpoint)", "zonly3080_mpc.png", (8.2, 4.4))
    fig_zonly("rof", "good-pool ROF (MA-7, mean of 3 metric seeds)", "zonly_rof.png", (8.2, 4.0))


if __name__ == "__main__":
    main()
