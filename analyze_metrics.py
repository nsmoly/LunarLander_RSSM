"""Comprehensive correlation analysis: metrics vs MPC performance.

Parses metrics log and MPC evaluation log, computes linear and quadratic
correlations (against MA-7 smoothed MPC mean return), and generates
multi-panel plots.

Supports both fixed-linearization (jac_) and time-varying (jac_dyn_) Jacobian
metrics.  Also computes two CROF (Composite Reward-Observability Fraction)
world-model scores.  CROF is built from the FIXED-linearization Jacobian
terms because they empirically correlate far more strongly with MPC than
their time-varying counterparts (see head-to-head analysis in the output
log).

The base ROF term is the *combined* good+bad reward observability,

    jac_rof_combined = 0.5 * jac_rof_good + 0.5 * jac_rof_bad

which empirically correlates better with MPC return than either
component alone (Spearman ~0.71 vs ~0.65 / ~0.69).  When jac_rof_bad
is unavailable in the log, CROF falls back to good-only jac_rof.

    crof_a = norm(jac_rof_combined)
           + 1.0 * (1 - norm(jac_ctrl_rank))
           + 1.0 * (1 - norm(jac_obs_rank))
           + 1.0 * norm(ol_obs_avg)

    crof_b = norm(jac_rof_combined)
           + 0.5 * (1 - norm(jac_ctrl_rank))
           + 0.5 * (1 - norm(jac_obs_rank))
           + 0.5 * norm(ol_obs_avg)

CROF-A uses fully-equal weights (all four terms weighted 1.0); CROF-B keeps
ROF as the primary signal with the three regularizers downweighted to 0.5.

Lower CROF = better checkpoint for MPC.
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =========================================================================
# Parsers
# =========================================================================
def parse_metrics(path):
    data = {}
    epoch = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"=== Epoch (\d+) ===", line)
            if m:
                epoch = int(m.group(1))
                data[epoch] = {}
                continue
            if epoch is None:
                continue
            for kv in re.finditer(r"(\w+)=([+-]?\d+\.?\d*(?:e[+-]?\d+)?|nan|inf|-inf)", line):
                key, val = kv.group(1), kv.group(2)
                data[epoch][key] = float(val)
    return data


def parse_mpc(path):
    data = {}
    epoch = None
    returns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)\s+--", line)
            if m:
                epoch = int(m.group(1))
                returns = []
                continue
            m = re.search(r"\[Episode \d+\] return=([+-]?[\d.]+)", line)
            if m and epoch is not None:
                returns.append(float(m.group(1)))
            m = re.search(r"mean_return=([+-]?[\d.]+)\s+worst_return=([+-]?[\d.]+)", line)
            if m and epoch is not None and returns:
                arr = np.array(returns)
                data[epoch] = {
                    "mean": float(m.group(1)),
                    "worst": float(m.group(2)),
                    "best": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                }
    return data


# =========================================================================
# Helpers
# =========================================================================
def ma(v, w=7):
    k = np.ones(w) / w
    p = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(p, k, mode="valid")[: len(v)]


def norm01(v):
    mn, mx = np.nanmin(v), np.nanmax(v)
    return (v - mn) / (mx - mn + 1e-12)


def corr_stats(x, y):
    r_p, _ = pearsonr(x, y)
    r_s, _ = spearmanr(x, y)
    coeffs = np.polyfit(x, y, 2)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r_p, r_s, r2, coeffs


# =========================================================================
# Analysis
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Metrics vs MPC correlation analysis")
    parser.add_argument("--metrics", default="metrics_eval_logs.txt")
    parser.add_argument("--mpc", default="mpc_eval_logs.txt")
    parser.add_argument("--log", default="metrics_analysis_logs.txt")
    args = parser.parse_args()

    log_file = open(args.log, "w", encoding="utf-8")

    def log(msg="", end="\n"):
        print(msg, end=end)
        log_file.write(msg + end)

    metrics = parse_metrics(args.metrics)
    mpc = parse_mpc(args.mpc)

    epochs = sorted(e for e in metrics if e in mpc and metrics[e])
    log("Matched epochs: {} (range {}-{})".format(len(epochs), epochs[0], epochs[-1]))
    log("Metrics file: {}".format(args.metrics))
    log("MPC file:     {}".format(args.mpc))

    mean_rtn = np.array([mpc[e]["mean"] for e in epochs])
    median_rtn = np.array([mpc[e]["median"] for e in epochs])
    best_rtn = np.array([mpc[e]["best"] for e in epochs])
    worst_rtn = np.array([mpc[e]["worst"] for e in epochs])
    sm_mean = ma(mean_rtn, 7)
    ep_arr = np.array(epochs)

    # Check which metric families are available
    sample = metrics[epochs[0]]
    has_jac_dyn = "jac_dyn_rof" in sample
    has_jac = "jac_rof" in sample

    metric_keys = [
        "val_loss", "val_kl", "val_recon",
        "post_obs_rmse", "post_rew_rmse",
        "ol_obs_start", "ol_obs_avg", "ol_obs_max", "ol_obs_end",
        "ol_rew_start", "ol_rew_avg", "ol_rew_max", "ol_rew_end",
        "ol_cumrew_err",
    ]
    if has_jac:
        metric_keys += [
            "jac_spec_radius", "jac_spec_radius_max",
            "jac_ctrl_rank", "jac_ctrl_cond", "jac_obs_rank", "jac_obs_cond",
            "jac_rcf", "jac_ocf", "jac_rof",
        ]
    if has_jac_dyn:
        metric_keys += [
            "jac_dyn_spec_radius", "jac_dyn_spec_radius_max",
            "jac_dyn_ctrl_rank", "jac_dyn_ctrl_cond", "jac_dyn_obs_rank", "jac_dyn_obs_cond",
            "jac_dyn_rcf", "jac_dyn_ocf", "jac_dyn_rof",
        ]
    has_rof_bad = "jac_rof_bad" in sample
    if has_rof_bad:
        metric_keys += ["jac_rof_bad", "jac_dyn_rof_bad"]
    metric_keys += ["emp_C", "emp_O", "emp_L"]

    def get_vals(key):
        return np.array([metrics[e].get(key, np.nan) for e in epochs])

    # =================================================================
    # ROF combined: alpha * jac_rof + (1-alpha) * jac_rof_bad
    # Final metric uses fixed alpha=0.5 (equal weight) for simplicity and
    # to avoid cherry-picking. Full sweep is logged for diagnostic purposes
    # (the metric is empirically flat between alpha=0.2 and alpha=0.6).
    # alpha = 1.0 -> jac_rof only; alpha = 0.0 -> jac_rof_bad only.
    # NOTE: This block must run BEFORE CROF computation so CROF can use
    # the combined ROF as its base term.
    # =================================================================
    ROF_COMBINED_ALPHA = 0.5
    rof_combined_vals = None
    if has_jac and has_rof_bad:
        rof_good = get_vals("jac_rof")
        rof_bad = get_vals("jac_rof_bad")
        sweep = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

        log("\n" + "=" * 90)
        log("ROF COMBINED ALPHA SWEEP: alpha*jac_rof + (1-alpha)*jac_rof_bad")
        log("(final metric uses alpha={:.2f})".format(ROF_COMBINED_ALPHA))
        log("=" * 90)
        log("{:>6s}  {:>9s}  {:>9s}  {:>9s}".format(
            "alpha", "Pearson", "Spearman", "R^2_quad"))
        log("-" * 90)

        best_abs_rho = -1.0
        best_alpha = None
        for alpha in sweep:
            combined = alpha * rof_good + (1 - alpha) * rof_bad
            mask_c = np.isfinite(combined) & np.isfinite(sm_mean)
            if mask_c.sum() < 10:
                continue
            r_p, r_s, r2, _ = corr_stats(combined[mask_c], sm_mean[mask_c])
            markers = []
            if abs(r_s) > best_abs_rho:
                best_abs_rho = abs(r_s)
                best_alpha = alpha
            if abs(alpha - ROF_COMBINED_ALPHA) < 1e-9:
                markers.append("<- final")
                rof_combined_vals = combined
            log("{:>6.2f}  {:+9.3f}  {:+9.3f}  {:>9.3f}  {}".format(
                alpha, r_p, r_s, r2, " ".join(markers)))

        if best_alpha is not None:
            log("\n(best-in-sweep alpha: {:.2f}, |Spearman|={:.3f}; "
                "final fixed at alpha={:.2f})".format(
                    best_alpha, best_abs_rho, ROF_COMBINED_ALPHA))

        if rof_combined_vals is not None:
            for e_idx, e in enumerate(epochs):
                metrics[e]["jac_rof_combined"] = rof_combined_vals[e_idx]
            metric_keys += ["jac_rof_combined"]

    # =================================================================
    # CROF scores (use FIXED linearization, built on combined ROF when available)
    # =================================================================
    crof_a = crof_b = None
    if has_jac:
        # CROF base: prefer jac_rof_combined (good+bad, alpha=0.5) when available;
        # fall back to jac_rof (good-only) for backward compatibility with old logs.
        if rof_combined_vals is not None:
            rof_base_key = "jac_rof_combined"
            t_rof = norm01(rof_combined_vals)
        else:
            rof_base_key = "jac_rof"
            t_rof = norm01(get_vals("jac_rof"))
        t_ctrl = 1 - norm01(get_vals("jac_ctrl_rank"))
        t_obs = 1 - norm01(get_vals("jac_obs_rank"))
        t_err = norm01(get_vals("ol_obs_avg"))

        # CROF-A: all terms weighted equally (1.0 each) — most parsimonious
        # CROF-B: ROF=1.0, regularizers=0.5 — reflects ROF as primary signal
        crof_a = t_rof + 1.0 * t_ctrl + 1.0 * t_obs + 1.0 * t_err
        crof_b = t_rof + 0.5 * t_ctrl + 0.5 * t_obs + 0.5 * t_err

        for e_idx, e in enumerate(epochs):
            metrics[e]["crof_a"] = crof_a[e_idx]
            metrics[e]["crof_b"] = crof_b[e_idx]
        metric_keys += ["crof_a", "crof_b"]
        log("\nCROF base ROF term: {}".format(rof_base_key))

    # =================================================================
    # Correlation table
    # =================================================================
    log("\n" + "=" * 90)
    log("CORRELATION TABLE: metric vs smoothed MPC mean (MA-7)")
    log("=" * 90)
    log("{:<26s} {:>8s} {:>8s} {:>8s}  {}".format(
        "Metric", "Pearson", "Spearman", "R^2_quad", "Direction"))
    log("-" * 90)

    corr_results = []
    for key in metric_keys:
        vals = get_vals(key)
        mask = np.isfinite(vals) & np.isfinite(sm_mean)
        if mask.sum() < 10:
            continue
        x, y = vals[mask], sm_mean[mask]
        r_p, r_s, r2_quad, coeffs = corr_stats(x, y)

        peak = "peak" if coeffs[0] < 0 else "valley"
        direction = "{} ({:.4f})".format(
            peak, -coeffs[1] / (2 * coeffs[0])) if abs(coeffs[0]) > 1e-10 else "linear"

        stars = " ***" if r2_quad > 0.15 else " **" if r2_quad > 0.08 else ""
        log("{:<26s} {:+8.3f} {:+8.3f} {:8.3f}  {}{}".format(
            key, r_p, r_s, r2_quad, direction, stars))
        corr_results.append((r2_quad, abs(r_p), r_p, r_s, r2_quad, key, vals))

    corr_results.sort(reverse=True)
    log("\n--- Top 15 by quadratic R^2 ---")
    for i, (_, _, r_p, r_s, r2, key, _) in enumerate(corr_results[:15]):
        log("  {:>2d}. {:<26s}  R^2_quad={:.3f}  r_pearson={:+.3f}  r_spearman={:+.3f}".format(
            i + 1, key, r2, r_p, r_s))

    # =================================================================
    # Head-to-head: jac_ vs jac_dyn_
    # =================================================================
    if has_jac and has_jac_dyn:
        log("\n" + "=" * 90)
        log("HEAD-TO-HEAD: jac_ (fixed) vs jac_dyn_ (time-varying)")
        log("=" * 90)
        log("{:<18s} {:>8s} {:>8s} {:>8s}  |  {:>8s} {:>8s} {:>8s}  | winner".format(
            "Metric", "jac_r", "jac_rho", "jac_R2q",
            "jac_dyn_r", "jac_dyn_rho", "jac_dyn_R2q"))
        log("-" * 90)
        for base in ["spec_radius", "spec_radius_max", "ctrl_rank", "ctrl_cond",
                     "obs_rank", "obs_cond", "rcf", "ocf", "rof"]:
            v_old = get_vals("jac_" + base)
            v_new = get_vals("jac_dyn_" + base)
            def _c(v):
                m = np.isfinite(v) & np.isfinite(sm_mean)
                if m.sum() < 10:
                    return 0, 0, 0
                return corr_stats(v[m], sm_mean[m])[:3]
            rp_o, rs_o, r2_o = _c(v_old)
            rp_n, rs_n, r2_n = _c(v_new)
            winner = "jac_dyn" if r2_n > r2_o else "jac" if r2_o > r2_n else "tie"
            log("{:<18s} {:+8.3f} {:+8.3f} {:8.3f}  |  {:+8.3f} {:+8.3f} {:8.3f}  | {}".format(
                base, rp_o, rs_o, r2_o, rp_n, rs_n, r2_n, winner))

    # =================================================================
    # Phase analysis
    # =================================================================
    log("\n" + "=" * 90)
    log("PHASE ANALYSIS: Pearson r within training phases")
    log("=" * 90)
    phases = [("Early 5-100", 5, 100), ("Mid 100-300", 100, 300), ("Late 300-500", 300, 500)]
    top_keys = [row[5] for row in corr_results[:12]]

    log("{:<26s}".format("Metric"), end="")
    for pname, _, _ in phases:
        log("  {:>14s}".format(pname), end="")
    log()
    log("-" * 90)
    for key in top_keys:
        vals = get_vals(key)
        log("{:<26s}".format(key), end="")
        for _, lo, hi in phases:
            pmask = np.array([(lo <= e < hi) for e in epochs])
            fmask = pmask & np.isfinite(vals)
            if fmask.sum() < 5:
                log("  {:>14s}".format("n/a"), end="")
            else:
                r = np.corrcoef(vals[fmask], mean_rtn[fmask])[0, 1]
                log("  {:>+14.3f}".format(r), end="")
        log()

    # =================================================================
    # Top/Bottom analysis
    # =================================================================
    log("\n" + "=" * 90)
    log("TOP-10 vs BOTTOM-10 MPC epochs")
    log("=" * 90)
    sorted_idx = np.argsort(mean_rtn)
    top10 = sorted_idx[-10:]
    bot10 = sorted_idx[:10]
    log("  Top-10 epochs: {}  (mean MPC: {:.1f})".format(
        [epochs[i] for i in top10], mean_rtn[top10].mean()))
    log("  Bot-10 epochs: {}  (mean MPC: {:.1f})".format(
        [epochs[i] for i in bot10], mean_rtn[bot10].mean()))
    log()
    for key in top_keys[:10]:
        vals = get_vals(key)
        t_mean = np.nanmean(vals[top10])
        b_mean = np.nanmean(vals[bot10])
        log("  {:<26s}  Top10={:.4f}  Bot10={:.4f}  diff={:+.4f}".format(
            key, t_mean, b_mean, t_mean - b_mean))

    # =================================================================
    # Checkpoint selection test
    # =================================================================
    log("\n" + "=" * 90)
    log("CHECKPOINT SELECTION TEST")
    log("=" * 90)
    log("Actual best MPC epoch: {} (mean return: {:.1f})".format(
        epochs[np.argmax(mean_rtn)], np.max(mean_rtn)))
    log("Actual best smoothed: {} (smoothed return: {:.1f})".format(
        epochs[np.argmax(sm_mean)], np.max(sm_mean)))
    log()

    selection_metrics = {
        "min ol_obs_end": ("ol_obs_end", "min"),
        "min ol_cumrew_err": ("ol_cumrew_err", "min"),
        "min ol_rew_avg": ("ol_rew_avg", "min"),
        "min val_loss": ("val_loss", "min"),
        "max emp_C": ("emp_C", "max"),
        "min post_rew_rmse": ("post_rew_rmse", "min"),
    }
    if has_jac:
        selection_metrics.update({
            "min jac_rof": ("jac_rof", "min"),
            "max jac_rcf": ("jac_rcf", "max"),
            "max jac_ctrl_rank": ("jac_ctrl_rank", "max"),
            "min crof_a": ("crof_a", "min"),
            "min crof_b": ("crof_b", "min"),
        })
    if has_jac and has_rof_bad and rof_combined_vals is not None:
        selection_metrics["min jac_rof_combined"] = ("jac_rof_combined", "min")
        selection_metrics["min jac_rof_bad"] = ("jac_rof_bad", "min")
    if has_jac_dyn:
        selection_metrics.update({
            "min jac_dyn_rof": ("jac_dyn_rof", "min"),
            "max jac_dyn_rcf": ("jac_dyn_rcf", "max"),
            "max jac_dyn_ctrl_rank": ("jac_dyn_ctrl_rank", "max"),
        })

    for desc, (key, mode) in selection_metrics.items():
        vals = get_vals(key)
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue
        if mode == "min":
            best_idx = np.argmin(np.where(mask, vals, np.inf))
        else:
            best_idx = np.argmax(np.where(mask, vals, -np.inf))
        sel_epoch = epochs[best_idx]
        sel_mpc = mpc[sel_epoch]["mean"]
        log("  {:<30s} -> epoch {:4d}  (MPC mean: {:+7.1f})".format(
            desc, sel_epoch, sel_mpc))

    # =====================================================================
    # PLOT 1: Comprehensive metrics overview
    # =====================================================================
    n_rows = 5
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 4 * n_rows + 2))
    fig.suptitle("World Model Metrics vs MPC Performance (LunarLander)",
                 fontsize=14, y=0.998)

    # P1: MPC Returns - same hump-emphasis treatment as the standalone
    #     mpc_performance.png: ylim sized to mean/median, min/max
    #     envelope is clipped.
    sm_median_p = ma(median_rtn, 7)
    sm_best_p = ma(best_rtn, 7)
    sm_worst_p = ma(worst_rtn, 7)
    ax = axes[0, 0]
    ax.fill_between(ep_arr, sm_worst_p, sm_best_p, alpha=0.10, color="blue",
                    label="Min--Max env (MA-7)")
    ax.plot(ep_arr, mean_rtn, "b-", alpha=0.2, label="Mean")
    ax.plot(ep_arr, sm_median_p, "g-", lw=1.3, alpha=0.7, label="Median (MA-7)")
    ax.plot(ep_arr, sm_mean, "b-", lw=2.5, label="Mean (MA-7)")
    env_lo_p = float(np.nanmin(sm_worst_p))
    env_hi_p = float(np.nanmax(sm_best_p))
    pad_p = max(10.0, 0.05 * (env_hi_p - env_lo_p))
    ax.set_ylim(env_lo_p - pad_p, env_hi_p + pad_p)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.axhline(y=0, color="gray", ls="--", alpha=0.4)
    ax.set_ylabel("MPC Return"); ax.set_title("MPC Performance")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P2: Multi-step Obs RMSE
    ax = axes[0, 1]
    ax.plot(ep_arr, get_vals("ol_obs_start"), "g-", alpha=0.7, label="start")
    ax.plot(ep_arr, get_vals("ol_obs_avg"), "b-", lw=2, label="avg")
    ax.plot(ep_arr, get_vals("ol_obs_max"), "r-", alpha=0.7, label="max")
    ax.plot(ep_arr, get_vals("ol_obs_end"), "r--", alpha=0.7, label="end")
    ax.set_ylabel("Obs RMSE"); ax.set_title("Open-Loop Obs RMSE")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P3: Open-loop Rew RMSE
    ax = axes[0, 2]
    ax.plot(ep_arr, get_vals("ol_rew_start"), "g-", alpha=0.7, label="start")
    ax.plot(ep_arr, get_vals("ol_rew_avg"), "b-", lw=2, label="avg")
    ax.plot(ep_arr, get_vals("ol_rew_max"), "r-", alpha=0.7, label="max")
    ax.plot(ep_arr, get_vals("ol_rew_end"), "r--", alpha=0.7, label="end")
    ax.set_ylabel("Rew RMSE"); ax.set_title("Open-Loop Rew RMSE")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P4: Cumulative reward error
    ax = axes[1, 0]
    ax.plot(ep_arr, get_vals("ol_cumrew_err"), "r-", alpha=0.7)
    ax.set_ylabel("Cum Reward Err"); ax.set_title("Open-Loop Cumulative Reward Error")
    ax.grid(True, alpha=0.3)

    # P5: Val loss + KL
    ax = axes[1, 1]
    ax.plot(ep_arr, get_vals("val_loss"), "b-", alpha=0.7, label="Val Loss")
    ax2 = ax.twinx()
    ax2.plot(ep_arr, get_vals("val_kl"), "r-", alpha=0.5, label="Val KL")
    ax.set_ylabel("Val Loss", color="blue"); ax2.set_ylabel("Val KL", color="red")
    ax.set_title("Validation Loss"); ax.grid(True, alpha=0.3)

    # P6: Empirical C and O
    ax = axes[1, 2]
    ax.plot(ep_arr, get_vals("emp_C"), "b-", label="emp_C")
    ax2 = ax.twinx()
    ax2.plot(ep_arr, get_vals("emp_O"), "r-", label="emp_O")
    ax.set_ylabel("C", color="blue"); ax2.set_ylabel("O", color="red")
    ax.set_title("Empirical C and O"); ax.grid(True, alpha=0.3)

    # P7: Spectral Radius
    ax = axes[2, 0]
    if has_jac:
        ax.plot(ep_arr, get_vals("jac_spec_radius"), "b-", label="jac (fixed)")
    if has_jac_dyn:
        ax.plot(ep_arr, get_vals("jac_dyn_spec_radius"), "r-", label="jac_dyn (TV)")
    ax.axhline(y=1.0, color="k", ls="--", alpha=0.3)
    ax.set_ylabel("Spectral Radius"); ax.set_title("Spectral Radius")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P8: Ctrl & Obs Rank
    ax = axes[2, 1]
    if has_jac:
        ax.plot(ep_arr, get_vals("jac_ctrl_rank"), "b-", alpha=0.7, label="jac ctrl")
        ax.plot(ep_arr, get_vals("jac_obs_rank"), "b--", alpha=0.7, label="jac obs")
    if has_jac_dyn:
        ax.plot(ep_arr, get_vals("jac_dyn_ctrl_rank"), "r-", alpha=0.7, label="jac_dyn ctrl")
        ax.plot(ep_arr, get_vals("jac_dyn_obs_rank"), "r--", alpha=0.7, label="jac_dyn obs")
    ax.set_ylabel("Effective Rank"); ax.set_title("Controllability & Observability Rank")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P9: ROF comparison (prefer combined ROF as primary)
    ax = axes[2, 2]
    if rof_combined_vals is not None:
        ax.plot(ep_arr, ma(rof_combined_vals, 7), "k-", lw=2,
                label="jac_rof_combined (MA-7)")
        ax.plot(ep_arr, rof_combined_vals, "k-", alpha=0.2)
    if has_jac:
        jac_rof = get_vals("jac_rof")
        ax.plot(ep_arr, ma(jac_rof, 7), "b-", lw=1.5, alpha=0.8,
                label="jac_rof good (MA-7)")
    if has_rof_bad:
        jac_rof_bad = get_vals("jac_rof_bad")
        ax.plot(ep_arr, ma(jac_rof_bad, 7), "r-", lw=1.5, alpha=0.8,
                label="jac_rof bad (MA-7)")
    if has_jac_dyn:
        jac_dyn_rof = get_vals("jac_dyn_rof")
        ax.plot(ep_arr, ma(jac_dyn_rof, 7), "g--", lw=1, alpha=0.6,
                label="jac_dyn_rof (MA-7)")
    ax.set_ylabel("ROF"); ax.set_title("ROF: Combined vs Components")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P10: Empirical metrics
    ax = axes[3, 0]
    ax.plot(ep_arr, ma(get_vals("emp_C"), 7), "b-", lw=2, label="emp_C (MA-7)")
    ax.plot(ep_arr, ma(get_vals("emp_O"), 7), "r-", lw=2, label="emp_O (MA-7)")
    ax.plot(ep_arr, ma(get_vals("emp_L"), 7), "g-", lw=2, label="emp_L (MA-7)")
    ax.set_ylabel("Value"); ax.set_title("Empirical C / O / L")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P11: Subspace alignment (prefer fixed Jacobian: stronger MPC predictor)
    ax = axes[3, 1]
    prefix = "jac" if has_jac else "jac_dyn"
    ax.plot(ep_arr, ma(get_vals(prefix + "_rcf"), 7), "b-", lw=2, label="RCF")
    ax.plot(ep_arr, ma(get_vals(prefix + "_ocf"), 7), "r-", lw=2, label="OCF")
    ax.plot(ep_arr, ma(get_vals(prefix + "_rof"), 7), "g-", lw=2, label="ROF")
    ax.set_ylabel("Fraction"); ax.set_title("Subspace Alignment ({})".format(prefix))
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # P12: CROF score (if available)
    ax = axes[3, 2]
    if crof_a is not None:
        comp_a_sm = ma(crof_a, 7)
        comp_b_sm = ma(crof_b, 7)
        ax.plot(ep_arr, sm_mean, "b-", lw=2, label="MPC Mean (MA-7)")
        ax.set_ylabel("MPC Return", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax2 = ax.twinx()
        ax2.plot(ep_arr, comp_a_sm, "r-", lw=2, label="CROF-A (MA-7)")
        ax2.plot(ep_arr, comp_b_sm, "r--", lw=2, label="CROF-B (MA-7)")
        ax2.set_ylabel("CROF (lower=better)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.invert_yaxis()
        sel_a = epochs[np.argmin(comp_a_sm)]
        sel_b = epochs[np.argmin(comp_b_sm)]
        ax.axvline(x=sel_a, color="red", ls="--", alpha=0.5,
                   label="Min CROF-A sm ({})".format(sel_a))
        ax.axvline(x=sel_b, color="red", ls=":", alpha=0.5,
                   label="Min CROF-B sm ({})".format(sel_b))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="lower left")
    else:
        ax.text(0.5, 0.5, "No jac_dyn_ metrics\navailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    ax.set_title("CROF vs MPC")
    ax.grid(True, alpha=0.3)

    # P13-P15: Scatter plots for the three core ROF building blocks
    # (jac_rof, jac_rof_bad, jac_dyn_rof). The composite metrics
    # (CROF-A, CROF-B, jac_rof_combined) have their own dedicated
    # figures (crof_analysis.png, rof_analysis.png) and are skipped here
    # to avoid duplication.
    scatter_keys = ["jac_rof", "jac_rof_bad", "jac_dyn_rof"]
    for idx, key in enumerate(scatter_keys):
        ax = axes[4, idx]
        vals = get_vals(key)
        mask = np.isfinite(vals) & np.isfinite(sm_mean)
        if mask.sum() < 3:
            ax.text(0.5, 0.5, "{}\n(no data)".format(key),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        x, y = vals[mask], sm_mean[mask]
        sc = ax.scatter(x, y, c=ep_arr[mask], cmap="viridis", s=20, alpha=0.7)
        r_p, r_s, r2, coeffs = corr_stats(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coeffs, xs), "r-", lw=2, alpha=0.7)
        ax.set_xlabel(key); ax.set_ylabel("MPC Mean Return (MA-7)")
        ax.set_title("{} (R2q={:.3f} rho={:+.3f})".format(key, r2, r_s))
        ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        if not ax.get_xlabel():
            ax.set_xlabel("Epoch", fontsize=8)

    plt.tight_layout()
    plt.savefig("metrics_vs_mpc_comprehensive.png", dpi=150, bbox_inches="tight")
    log("\nSaved: metrics_vs_mpc_comprehensive.png")

    # =====================================================================
    # STANDALONE MPC FIGURE (paper/figures/mpc_performance.png)
    # Y-axis is tightened to the smoothed-mean envelope so the hump is
    # pronounced. Median / min / max are also smoothed (MA-7) so they fit
    # inside the panel as a clean envelope rather than spiky raw lines.
    # =====================================================================
    sm_median = ma(median_rtn, 7)
    sm_best = ma(best_rtn, 7)
    sm_worst = ma(worst_rtn, 7)

    fig_mpc, ax_mpc = plt.subplots(figsize=(7, 5.5))
    # Wide min/max envelope sits behind the curves. Its extreme range gets
    # clipped by the tight ylim below; that is intentional - the panel is
    # framed around the actual hump, not around outliers.
    ax_mpc.fill_between(ep_arr, sm_worst, sm_best, alpha=0.10, color="blue",
                        label="Min--Max envelope (MA-7, clipped)")
    ax_mpc.plot(ep_arr, mean_rtn, "b-", alpha=0.20, label="Mean return")
    ax_mpc.plot(ep_arr, sm_median, "g-", lw=1.5, alpha=0.7,
                label="Median return (MA-7)")
    ax_mpc.plot(ep_arr, sm_mean, "b-", lw=3.0, label="Mean return (MA-7)")
    best_sm_idx = int(np.argmax(sm_mean))
    ax_mpc.axvline(x=epochs[best_sm_idx], color="red", ls="--", alpha=0.65,
                   label="Best smoothed (epoch {})".format(epochs[best_sm_idx]))
    ax_mpc.axhline(y=0, color="gray", ls="--", alpha=0.4)

    # ylim auto-fit to the smoothed (MA-7) min/max envelope plus a small
    # pad. This makes the panel adapt to whatever MPC sweep is supplied,
    # while keeping the mean MA-7 hump clearly readable.
    env_lo = float(np.nanmin(sm_worst))
    env_hi = float(np.nanmax(sm_best))
    pad = max(10.0, 0.05 * (env_hi - env_lo))
    ax_mpc.set_ylim(env_lo - pad, env_hi + pad)
    ax_mpc.set_xlim(epochs[0], epochs[-1])

    ax_mpc.set_xlabel("Training Epoch")
    ax_mpc.set_ylabel("MPC Return (20 episodes)")
    ax_mpc.set_title("CEM-MPC Performance Over World Model Training")
    ax_mpc.legend(fontsize=9, loc="lower left")
    ax_mpc.grid(True, alpha=0.3)
    fig_mpc.tight_layout()
    plt.savefig("mpc_performance.png", dpi=150, bbox_inches="tight")
    log("Saved: mpc_performance.png  "
        "(MA-7 mean [{:.1f}, {:.1f}], median MA-7 [{:.1f}, {:.1f}], "
        "ylim [{:.1f}, {:.1f}])"
        .format(float(np.nanmin(sm_mean)), float(np.nanmax(sm_mean)),
                float(np.nanmin(sm_median)), float(np.nanmax(sm_median)),
                env_lo - pad, env_hi + pad))
    plt.close(fig_mpc)

    # =====================================================================
    # PLOT 2: ROF analysis (primary = combined good+bad if available)
    # =====================================================================
    # Preference order:
    #  1. jac_rof_combined  (alpha=0.5 mix of good+bad; strongest MPC predictor)
    #  2. jac_rof           (good-only fixed Jacobian; backward compat)
    #  3. jac_dyn_rof       (time-varying; weakest, only if no fixed available)
    if rof_combined_vals is not None:
        rof_key = "jac_rof_combined"
    elif has_jac:
        rof_key = "jac_rof"
    else:
        rof_key = "jac_dyn_rof"
    rof_vals = get_vals(rof_key)
    rof_sm = ma(rof_vals, 7)
    mask_rof = np.isfinite(rof_vals) & np.isfinite(sm_mean)
    _, rho_rof, r2_rof, coeff_rof = corr_stats(rof_vals[mask_rof], sm_mean[mask_rof])

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle("ROF: Reward Observability Fraction Analysis", fontsize=14)

    # Panel 1: Time series overlay (primary ROF)
    ax = axes2[0, 0]
    ax.plot(ep_arr, mean_rtn, "b-", alpha=0.2)
    ax.plot(ep_arr, sm_mean, "b-", lw=2, label="MPC Mean (MA-7)")
    ax.set_ylabel("MPC Return", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2 = ax.twinx()
    ax2.plot(ep_arr, rof_vals, "r-", alpha=0.2)
    ax2.plot(ep_arr, rof_sm, "r-", lw=2, label="{} (MA-7)".format(rof_key))
    ax2.set_ylabel("ROF", color="red"); ax2.tick_params(axis="y", labelcolor="red")
    ax2.invert_yaxis()
    best_sm_idx = np.argmax(sm_mean)
    min_rof_idx = np.argmin(rof_sm)
    ax.axvline(x=epochs[best_sm_idx], color="blue", ls="--", alpha=0.5,
               label="Best MPC sm ({})".format(epochs[best_sm_idx]))
    ax.axvline(x=epochs[min_rof_idx], color="red", ls="--", alpha=0.5,
               label="Min {} sm ({})".format(rof_key, epochs[min_rof_idx]))
    ax.set_xlabel("Epoch")
    ax.set_title("MPC vs {} (inverted)\nrho={:+.3f}  R2q={:.3f}".format(
        rof_key, rho_rof, r2_rof))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Panel 2: Scatter with quadratic fit
    ax = axes2[0, 1]
    x_rof, y_rof = rof_vals[mask_rof], sm_mean[mask_rof]
    sc = ax.scatter(x_rof, y_rof, c=ep_arr[mask_rof], cmap="viridis", s=25, alpha=0.7,
                    edgecolors="k", linewidths=0.3)
    xs = np.linspace(x_rof.min(), x_rof.max(), 100)
    ax.plot(xs, np.polyval(coeff_rof, xs), "r-", lw=2)
    ax.set_xlabel(rof_key); ax.set_ylabel("MPC Mean Return (MA-7)")
    ax.set_title("{} scatter (R2q={:.3f} rho={:+.3f})".format(rof_key, r2_rof, rho_rof))
    plt.colorbar(sc, ax=ax, label="Epoch"); ax.grid(True, alpha=0.3)

    # Panel 3: CROF checkpoint selection
    ax = axes2[1, 0]
    ax.plot(ep_arr, sm_mean, "b-", lw=2, label="MPC Mean (MA-7)")
    ax.axhline(y=0, color="k", ls="--", alpha=0.2)
    ax.axvline(x=epochs[best_sm_idx], color="blue", ls="--", alpha=0.5,
               label="Best MPC sm ({})".format(epochs[best_sm_idx]))
    ax.axvline(x=epochs[min_rof_idx], color="red", ls="--", alpha=0.5,
               label="Min {} sm ({})".format(rof_key, epochs[min_rof_idx]))
    if crof_a is not None:
        comp_a_sm = ma(crof_a, 7)
        comp_b_sm = ma(crof_b, 7)
        ax.axvline(x=epochs[np.argmin(comp_a_sm)], color="green", ls="--", alpha=0.5,
                   label="Min CROF-A sm ({})".format(epochs[np.argmin(comp_a_sm)]))
        ax.axvline(x=epochs[np.argmin(comp_b_sm)], color="purple", ls="--", alpha=0.5,
                   label="Min CROF-B sm ({})".format(epochs[np.argmin(comp_b_sm)]))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Smoothed MPC Return")
    ax.set_title("Checkpoint Selection Comparison")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 4: All subspace alignment metrics (prefer fixed)
    ax = axes2[1, 1]
    prefix = "jac" if has_jac else "jac_dyn"
    ax.plot(ep_arr, ma(get_vals(prefix + "_rof"), 7), "g-", lw=2, label="ROF (MA-7)")
    ax.plot(ep_arr, ma(get_vals(prefix + "_rcf"), 7), "b-", lw=2, label="RCF (MA-7)")
    ax.plot(ep_arr, ma(get_vals(prefix + "_ocf"), 7), "r-", lw=2, label="OCF (MA-7)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction")
    ax.set_title("Subspace Alignment Metrics ({})".format(prefix))
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rof_analysis.png", dpi=150, bbox_inches="tight")
    log("Saved: rof_analysis.png")

    # =====================================================================
    # PLOT 3: CROF score analysis
    # =====================================================================
    if crof_a is not None:
        comp_a_sm = ma(crof_a, 7)
        comp_b_sm = ma(crof_b, 7)
        sel_a_idx = np.argmin(comp_a_sm)
        sel_b_idx = np.argmin(comp_b_sm)
        sel_a_ep = epochs[sel_a_idx]
        sel_b_ep = epochs[sel_b_idx]
        best_mpc_ep = epochs[np.argmax(sm_mean)]

        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
        fig3.suptitle("CROF: Checkpoint Selection Analysis", fontsize=14)

        for col, (label, comp_raw, comp_sm, sel_ep, sel_idx, weight) in enumerate([
            ("CROF-A", crof_a, comp_a_sm, sel_a_ep, sel_a_idx, "1.0"),
            ("CROF-B", crof_b, comp_b_sm, sel_b_ep, sel_b_idx, "0.5"),
        ]):
            # Top row: time-series overlay
            ax = axes3[0, col]
            ax.plot(ep_arr, mean_rtn, "b-", alpha=0.15)
            ax.plot(ep_arr, sm_mean, "b-", lw=2, label="MPC Mean (MA-7)")
            ax.set_ylabel("MPC Return", color="blue")
            ax.tick_params(axis="y", labelcolor="blue")
            ax_r = ax.twinx()
            ax_r.plot(ep_arr, comp_raw, "r-", alpha=0.15)
            ax_r.plot(ep_arr, comp_sm, "r-", lw=2, label="{} (MA-7)".format(label))
            ax_r.set_ylabel("{} (lower=better)".format(label), color="red")
            ax_r.tick_params(axis="y", labelcolor="red")
            ax_r.invert_yaxis()
            ax.axvline(x=best_mpc_ep, color="blue", ls="--", alpha=0.6,
                       label="Best MPC sm ({})".format(best_mpc_ep))
            ax.axvline(x=sel_ep, color="red", ls="--", alpha=0.6,
                       label="Min {} sm ({})".format(label, sel_ep))
            raw_sel_idx = int(np.nanargmin(comp_raw))
            raw_sel_ep = epochs[raw_sel_idx]
            ax.axvline(x=raw_sel_ep, color="red", ls=":", alpha=0.6,
                       label="Min {} raw ({})".format(label, raw_sel_ep))
            gap = abs(sel_ep - best_mpc_ep)
            mpc_at_sel = mpc[sel_ep]["mean"]
            ax.set_xlabel("Epoch")
            mask_c = np.isfinite(comp_raw) & np.isfinite(sm_mean)
            rho_c = spearmanr(comp_raw[mask_c], sm_mean[mask_c])[0]
            _, _, r2_c, _ = corr_stats(comp_raw[mask_c], sm_mean[mask_c])
            ax.set_title("{} (ROF+{}*ctrl+{}*obs+{}*err)\n"
                         "Selected: ep {} (gap={}, MPC={:.0f})  "
                         "rho={:+.3f}  R2q={:.3f}".format(
                             label, weight, weight, weight,
                             sel_ep, gap, mpc_at_sel, rho_c, r2_c))
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_r.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
            ax.grid(True, alpha=0.3)

            # Bottom row: scatter with both linear and quadratic fits.
            # CROF is monotonic ("lower=better"), so the linear fit is the
            # interpretable trend. The quadratic fit is shown as a sanity
            # check — its valley near the right edge is a regression artifact
            # driven by a few high-CROF early-epoch outliers and should NOT
            # be read as "best MPC at intermediate CROF".
            ax = axes3[1, col]
            x_c, y_c = comp_raw[mask_c], sm_mean[mask_c]
            sc = ax.scatter(x_c, y_c, c=ep_arr[mask_c], cmap="viridis", s=25,
                            alpha=0.7, edgecolors="k", linewidths=0.3)
            _, _, r2_c, coeffs_c = corr_stats(x_c, y_c)
            xs = np.linspace(x_c.min(), x_c.max(), 100)
            # Linear fit (primary trend for monotonic predictor)
            lin_coeffs = np.polyfit(x_c, y_c, 1)
            y_lin_pred = np.polyval(lin_coeffs, x_c)
            ss_res = np.sum((y_c - y_lin_pred) ** 2)
            ss_tot = np.sum((y_c - y_c.mean()) ** 2)
            r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.plot(xs, np.polyval(lin_coeffs, xs), "g-", lw=2.5, alpha=0.85,
                    label="Linear fit (R2={:.3f})".format(r2_lin))
            ax.plot(xs, np.polyval(coeffs_c, xs), "r--", lw=1.5, alpha=0.6,
                    label="Quadratic fit (R2={:.3f})".format(r2_c))
            ax.set_xlabel("{} (lower = better)".format(label))
            ax.set_ylabel("MPC Mean Return (MA-7)")
            ax.set_title("{} scatter  rho={:+.3f}".format(label, rho_c))
            ax.legend(fontsize=7, loc="upper right")
            plt.colorbar(sc, ax=ax, label="Epoch")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("crof_analysis.png", dpi=150, bbox_inches="tight")
        log("Saved: crof_analysis.png")

    # =====================================================================
    # PLOT 4: Correlation bar chart — metrics with |rho| >= threshold
    # (full ranking is reported in the correlation table in the paper).
    # =====================================================================
    RHO_THRESHOLD = 0.10
    sorted_corr_all = sorted(corr_results, key=lambda r: abs(r[3]), reverse=True)
    n_total = len(sorted_corr_all)
    sorted_corr = [r for r in sorted_corr_all if abs(r[3]) >= RHO_THRESHOLD]
    n_omitted = n_total - len(sorted_corr)
    bar_keys = [r[5] for r in sorted_corr]
    bar_rho = [abs(r[3]) for r in sorted_corr]
    bar_r2 = [r[4] for r in sorted_corr]
    n_bars = len(bar_keys)
    bar_h = 0.35

    fig4, ax4 = plt.subplots(figsize=(11, max(6, n_bars * 0.40 + 2)))
    ax4.set_title(
        "Metrics vs MA-7 Smoothed MPC Mean Return Correlation Strength "
        "(showing {} of {} metrics with $|\\rho| \\geq {:.2f}$)"
        .format(n_bars, n_total, RHO_THRESHOLD),
        fontsize=12, pad=10)

    y_pos = np.arange(n_bars)
    colors_rho = ["#4CAF50" if v >= 0.6 else "#A5D6A7" if v >= 0.3 else "#E0E0E0"
                  for v in bar_rho]
    colors_r2 = ["#2196F3" if v >= 0.15 else "#90CAF9" if v >= 0.08 else "#E0E0E0"
                 for v in bar_r2]

    ax4.barh(y_pos - bar_h / 2, bar_rho, height=bar_h, color=colors_rho,
             edgecolor="grey", linewidth=0.3, label="|Spearman rho|")
    ax4.barh(y_pos + bar_h / 2, bar_r2, height=bar_h, color=colors_r2,
             edgecolor="grey", linewidth=0.3, label="R^2_quad")

    for i, (rho_v, r2_v) in enumerate(zip(bar_rho, bar_r2)):
        ax4.text(rho_v + 0.005, i - bar_h / 2, "{:.3f}".format(rho_v),
                 va="center", fontsize=7, color="#2E7D32")
        ax4.text(r2_v + 0.005, i + bar_h / 2, "{:.3f}".format(r2_v),
                 va="center", fontsize=7, color="#1565C0")

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(bar_keys, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Correlation Strength")
    ax4.axvline(x=0.6, color="#2E7D32", ls="--", alpha=0.4, label="rho: strong (0.6)")
    ax4.axvline(x=0.3, color="#66BB6A", ls="--", alpha=0.4, label="rho: moderate (0.3)")
    ax4.axvline(x=0.15, color="#1565C0", ls="--", alpha=0.4, label="R^2: strong (0.15)")
    ax4.legend(fontsize=7, loc="lower right")
    ax4.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig("correlation_barchart.png", dpi=150, bbox_inches="tight")
    log("Saved: correlation_barchart.png  (showing {} of {} metrics, "
        "{} omitted with |rho| < {})"
        .format(n_bars, n_total, n_omitted, RHO_THRESHOLD))

    # =====================================================================
    # CROF score summary
    # =====================================================================
    if crof_a is not None:
        comp_a_sm = ma(crof_a, 7)
        comp_b_sm = ma(crof_b, 7)
        log("\n" + "=" * 90)
        log("CROF SCORE SUMMARY")
        log("=" * 90)
        for label, comp, comp_sm in [
            ("CROF-A (ROF+1.0*ctrl+1.0*obs+1.0*err)", crof_a, comp_a_sm),
            ("CROF-B (ROF+0.5*ctrl+0.5*obs+0.5*err)", crof_b, comp_b_sm),
        ]:
            sel_idx = np.argmin(comp_sm)
            sel_ep = epochs[sel_idx]
            mask = np.isfinite(comp) & np.isfinite(sm_mean)
            rho, _ = spearmanr(comp[mask], sm_mean[mask])
            _, _, r2, _ = corr_stats(comp[mask], sm_mean[mask])
            log("  {:<45s}".format(label))
            log("    Selected epoch: {:4d}  (MPC mean: {:+.1f})".format(
                sel_ep, mpc[sel_ep]["mean"]))
            log("    Spearman rho: {:+.3f}   R^2_quad: {:.3f}".format(rho, r2))

    # =====================================================================
    # Sliding window Spearman
    # =====================================================================
    log("\n" + "=" * 90)
    log("SLIDING WINDOW SPEARMAN (window=6 checkpoints)")
    log("=" * 90)
    window_size = 6
    rof_for_window = get_vals(rof_key)
    cumrew = get_vals("ol_cumrew_err")
    ol_end = get_vals("ol_obs_end")
    log("{:<10s} {:>12s} {:>12s} {:>10s}".format(
        "Center", rof_key[:10] + "_rho", "ol_end_rho", "cumrew_rho"))
    for start in range(0, len(epochs) - window_size + 1, 3):
        end = start + window_size
        w_epochs = epochs[start:end]
        w_rtn = mean_rtn[start:end]
        w_rof = rof_for_window[start:end]
        w_ol_end = ol_end[start:end]
        w_cum = cumrew[start:end]
        center = (w_epochs[0] + w_epochs[-1]) // 2
        r1, _ = spearmanr(w_rof, w_rtn)
        m_end = np.isfinite(w_ol_end)
        r2 = spearmanr(w_ol_end[m_end], w_rtn[m_end])[0] if m_end.sum() >= 4 else float("nan")
        r3, _ = spearmanr(w_cum, w_rtn)
        log("  {:>6d}    {:>+10.3f} {:>+10.3f} {:>+10.3f}".format(center, r1, r2, r3))

    # =====================================================================
    # CROF per checkpoint table
    # =====================================================================
    if crof_a is not None:
        log("\n" + "=" * 90)
        log("CROF SCORES PER CHECKPOINT")
        log("=" * 90)
        log("{:>8s}  {:>12s}  {:>12s}  {:>12s}".format(
            "Epoch", "Score_A", "Score_B", "MPC_mean"))
        log("-" * 50)
        for i, e in enumerate(epochs):
            log("{:>8d}  {:>12.4f}  {:>12.4f}  {:>+12.1f}".format(
                e, crof_a[i], crof_b[i], mpc[e]["mean"]))

    log_file.close()
    log_file = None
    print("Saved: {}".format(args.log))


if __name__ == "__main__":
    main()
