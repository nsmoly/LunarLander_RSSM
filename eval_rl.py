# ===========================================================================
# Batch RL policy evaluation across actor-critic checkpoints.
# Runs test_policy.py for each actor checkpoint and aggregates logs.
#
# Supports both world-model-based AC (--actor_type latent) and model-free AC
# (--actor_type obs). For latent, --world_model is required and held fixed
# across the sweep.
#
# Usage examples:
#
#   # New WM-based AC (trained with WM epoch 295)
#   python eval_rl.py \
#       --checkpoints_dir archive/V4_good_landing_extendedTrainingBest_04022026/checkpoints_actorcritic \
#       --actor_type latent \
#       --world_model archive/V4_good_landing_extendedTrainingBest_04022026/checkpoints_worldmodel/world_model_20260331_092643_epoch_295.pt \
#       --output rl_eval_logs_newAC.txt
#
#   # Old WM-based AC (trained with WM epoch 200)
#   python eval_rl.py \
#       --checkpoints_dir archive/V4_good_landing_afterRSSMFixes_03182026/checkpoints_ac_v4_AfterRSSMFixes_03182026 \
#       --actor_type latent \
#       --world_model archive/V4_good_landing_afterRSSMFixes_03182026/checkpoints_wm_v4_AfterRSSMFixes_03182026/world_model_20260318_105750_epoch_200.pt \
#       --output rl_eval_logs_oldAC.txt
#
#   # Model-free AC
#   python eval_rl.py \
#       --checkpoints_dir archive/V4_good_landing_afterRSSMFixes_03182026/checkpoints_acmf_v4_modelfreeAdd_03262026 \
#       --actor_type obs \
#       --output rl_eval_logs_mfAC.txt
# ===========================================================================

import argparse
import os
import re
import subprocess
import sys
import time


SCORECARD_LINE_1_RE = re.compile(
    r"\[Run\]\s+mean_return=([+-]?[\d.]+)\s+"
    r"worst_return=([+-]?[\d.]+)\s+"
    r"main_action_mix=([\d.]+)%\s+"
    r"main_action_mix_air=([\d.]+)%\s+"
    r"near_avg_abs_angle=([\d.nan]+)\s+"
    r"near_avg_down_speed=([\d.nan]+)\s+"
    r"touchdown_median_abs_ang_vel=([\d.nan]+)\s+"
    r"touchdown_p90_abs_ang_vel=([\d.nan]+)"
)
SCORECARD_LINE_2_RE = re.compile(
    r"\[Run\]\s+perfect=(\d+)\s+negative=(\d+)\s+catastrophic=(\d+)\s+"
    r"avg_entropy=([\d.]+)\s+avg_steps=([\d.]+)"
)


def parse_scorecard(stdout: str):
    """Extract scorecard fields from test_policy.py stdout. Returns dict or None."""
    m1 = SCORECARD_LINE_1_RE.search(stdout)
    m2 = SCORECARD_LINE_2_RE.search(stdout)
    if not m1 or not m2:
        return None
    try:
        return {
            "mean_return": float(m1.group(1)),
            "worst_return": float(m1.group(2)),
            "main_action_mix": float(m1.group(3)),
            "main_action_mix_air": float(m1.group(4)),
            "near_avg_abs_angle": float(m1.group(5)),
            "near_avg_down_speed": float(m1.group(6)),
            "touchdown_median_abs_ang_vel": float(m1.group(7)),
            "touchdown_p90_abs_ang_vel": float(m1.group(8)),
            "perfect": int(m2.group(1)),
            "negative": int(m2.group(2)),
            "catastrophic": int(m2.group(3)),
            "avg_entropy": float(m2.group(4)),
            "avg_steps": float(m2.group(5)),
        }
    except (ValueError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run AC policy evaluation on every actor checkpoint in a folder"
    )
    parser.add_argument("--checkpoints_dir", required=True,
                        help="Folder containing actor checkpoints")
    parser.add_argument("--actor_type", choices=["latent", "obs"], required=True,
                        help="'latent' = WM-based AC (needs --world_model); "
                             "'obs' = model-free AC")
    parser.add_argument("--world_model", default=None,
                        help="World model checkpoint (required for --actor_type latent)")
    parser.add_argument("--actor_pattern", default=r"actor.*epoch_(\d+)\.pt$",
                        help="Regex matching actor filenames; the first capture group "
                             "must be the integer epoch number "
                             r"(default: 'actor.*epoch_(\d+)\.pt$').")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per checkpoint (default: 20)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Max steps per episode (default: 600, matches MPC sweep)")
    parser.add_argument("--epoch_min", type=int, default=None,
                        help="Only evaluate checkpoints with epoch >= this value")
    parser.add_argument("--epoch_max", type=int, default=None,
                        help="Only evaluate checkpoints with epoch <= this value")
    parser.add_argument("--epoch_stride", type=int, default=1,
                        help="Take every N-th checkpoint (default: 1; "
                             "use 5 for model-free actors saved every 10 epochs)")
    parser.add_argument("--output", default="rl_eval_logs.txt",
                        help="Output log file (default: rl_eval_logs.txt)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Print top-K checkpoints by mean_return at the end (default: 10)")
    args = parser.parse_args()

    if args.actor_type == "latent" and not args.world_model:
        parser.error("--world_model is required when --actor_type=latent")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(script_dir, "test_policy.py")

    ckpt_dir = args.checkpoints_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(script_dir, ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: checkpoints_dir does not exist: {ckpt_dir}")
        sys.exit(1)

    if args.actor_type == "latent":
        wm_path = args.world_model
        if not os.path.isabs(wm_path):
            wm_path = os.path.join(script_dir, wm_path)
        if not os.path.isfile(wm_path):
            print(f"ERROR: world_model checkpoint does not exist: {wm_path}")
            sys.exit(1)
    else:
        wm_path = None

    actor_re = re.compile(args.actor_pattern)
    checkpoints = []
    for f in sorted(os.listdir(ckpt_dir)):
        m = actor_re.search(f)
        if m:
            try:
                epoch = int(m.group(1))
                checkpoints.append((epoch, f))
            except (ValueError, IndexError):
                continue
    checkpoints.sort(key=lambda x: x[0])

    if args.epoch_min is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e >= args.epoch_min]
    if args.epoch_max is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e <= args.epoch_max]

    if args.epoch_stride > 1:
        checkpoints = checkpoints[::args.epoch_stride]

    if not checkpoints:
        print(f"No matching actor files found in {ckpt_dir} "
              f"(pattern: {args.actor_pattern})")
        return

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(script_dir, out_path)

    print(f"Found {len(checkpoints)} checkpoints (after filters)")
    print(f"Actor type:    {args.actor_type}")
    print(f"World model:   {wm_path or '(none, model-free)'}")
    print(f"Episodes:      {args.episodes}, seed={args.seed}, max_steps={args.max_steps}")
    print(f"Output:        {out_path}\n")

    file_mode = "a" if args.append else "w"
    sweep_records = []

    with open(out_path, file_mode, encoding="utf-8") as log:
        epoch_range = ""
        if args.epoch_min is not None or args.epoch_max is not None:
            epoch_range = f", epoch range: [{args.epoch_min or '*'}-{args.epoch_max or '*'}]"

        header = (
            f"\n{'=' * 70}\n"
            f"AC Policy Evaluation Sweep\n"
            f"Actor type: {args.actor_type}\n"
            f"Checkpoints dir: {ckpt_dir}\n"
            f"World model: {wm_path or '(none, model-free)'}\n"
            f"Checkpoints: {len(checkpoints)}, "
            f"Episodes per checkpoint: {args.episodes}, Seed: {args.seed}{epoch_range}\n"
            f"Parameters: max_steps={args.max_steps}, "
            f"epoch_stride={args.epoch_stride}, deterministic=True\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 70}\n\n"
        )
        log.write(header)
        log.flush()
        print(header, end="")

        total_t0 = time.time()
        for i, (epoch, ckpt_name) in enumerate(checkpoints):
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            block_header = (
                f"{'=' * 70}\n"
                f"  [{i + 1}/{len(checkpoints)}] Actor epoch {epoch} -- {ckpt_name}\n"
                f"{'=' * 70}\n"
            )
            print(block_header, end="")
            log.write(block_header)
            log.flush()

            t0 = time.time()
            cmd = [
                sys.executable, test_script,
                "--actor_type", args.actor_type,
                "--actor", ckpt_path,
                "--episodes", str(args.episodes),
                "--seed", str(args.seed),
                "--max_steps", str(args.max_steps),
                "--config", args.config,
            ]
            if args.actor_type == "latent":
                cmd.extend(["--world_model", wm_path])

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=script_dir,
            )
            elapsed = time.time() - t0
            body = result.stdout
            if result.returncode != 0:
                body += (
                    f"\n--- STDERR (exit code {result.returncode}) ---\n"
                    f"{result.stderr}\n"
                )
            body += f"[Eval time: {elapsed:.1f}s]\n\n"

            print(body, end="")
            log.write(body)
            log.flush()

            scorecard = parse_scorecard(result.stdout)
            if scorecard is not None:
                scorecard["epoch"] = epoch
                scorecard["ckpt"] = ckpt_name
                scorecard["eval_seconds"] = elapsed
                sweep_records.append(scorecard)

        total_elapsed = time.time() - total_t0

        # ---- Sweep summary tables ----
        if sweep_records:
            summary = ["\n" + "=" * 70 + "\n",
                       "Sweep summary (sorted by epoch)\n",
                       "=" * 70 + "\n"]
            hdr = (
                f"{'epoch':>6}  {'mean':>8}  {'worst':>8}  "
                f"{'perfect':>7}  {'neg':>3}  {'catas':>5}  "
                f"{'avg_steps':>9}  {'avg_ent':>7}\n"
            )
            summary.append(hdr)
            for rec in sorted(sweep_records, key=lambda r: r["epoch"]):
                summary.append(
                    f"{rec['epoch']:>6}  {rec['mean_return']:+8.2f}  {rec['worst_return']:+8.2f}  "
                    f"{rec['perfect']:>7}  {rec['negative']:>3}  {rec['catastrophic']:>5}  "
                    f"{rec['avg_steps']:>9.1f}  {rec['avg_entropy']:>7.3f}\n"
                )

            top_k = min(args.top_k, len(sweep_records))
            summary.append("\n" + "=" * 70 + "\n")
            summary.append(f"Top {top_k} checkpoints by mean_return\n")
            summary.append("=" * 70 + "\n")
            summary.append(hdr)
            top = sorted(sweep_records, key=lambda r: r["mean_return"], reverse=True)[:top_k]
            for rec in top:
                summary.append(
                    f"{rec['epoch']:>6}  {rec['mean_return']:+8.2f}  {rec['worst_return']:+8.2f}  "
                    f"{rec['perfect']:>7}  {rec['negative']:>3}  {rec['catastrophic']:>5}  "
                    f"{rec['avg_steps']:>9.1f}  {rec['avg_entropy']:>7.3f}\n"
                )

            best = top[0]
            summary.append(
                f"\nBest by mean_return: epoch {best['epoch']} "
                f"({best['ckpt']}) -> mean={best['mean_return']:+.2f}, "
                f"worst={best['worst_return']:+.2f}, "
                f"perfect={best['perfect']}/{args.episodes}, "
                f"catastrophic={best['catastrophic']}/{args.episodes}\n"
            )

            summary_text = "".join(summary)
            log.write(summary_text)
            print(summary_text, end="")

        footer = (
            f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(total: {total_elapsed / 60:.1f} min, "
            f"{len(sweep_records)}/{len(checkpoints)} checkpoints scored)\n"
        )
        log.write(footer)
        print(footer, end="")

    print(f"\nAll done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
