# ===========================================================================
# Batch MPC (CEM) evaluation of all world model checkpoints.
# Runs wm_mpc_policy.py for each checkpoint and collects results into one log.
#
# Usage:
#     python eval_mpc.py [--episodes N] [--seed S]
# ===========================================================================

import subprocess
import sys
import os
import re
import time
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run CEM MPC evaluation on every world-model checkpoint"
    )
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per checkpoint (default: 20)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--output", default="mpc_eval_logs.txt")
    parser.add_argument("--epoch_min", type=int, default=None,
                        help="Only evaluate checkpoints with epoch >= this value")
    parser.add_argument("--epoch_max", type=int, default=None,
                        help="Only evaluate checkpoints with epoch <= this value")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    parser.add_argument("--epoch_stride", type=int, default=1,
                        help="Take every N-th checkpoint (1=all, 2=every other, etc.)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mpc_script = os.path.join(script_dir, "wm_mpc_policy.py")
    ckpt_dir = os.path.join(script_dir, args.checkpoints_dir)

    checkpoints = []
    for f in os.listdir(ckpt_dir):
        m = re.search(r"epoch_(\d+)\.pt$", f)
        if m and f.startswith("world_model_"):
            checkpoints.append((int(m.group(1)), f))
    checkpoints.sort()

    if args.epoch_min is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e >= args.epoch_min]
    if args.epoch_max is not None:
        checkpoints = [(e, f) for e, f in checkpoints if e <= args.epoch_max]

    if args.epoch_stride > 1:
        checkpoints = checkpoints[::args.epoch_stride]

    if not checkpoints:
        print(f"No matching world_model_*_epoch_*.pt files found in {ckpt_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Running {args.episodes} episodes per checkpoint, seed={args.seed}")
    print(f"Output: {args.output}\n")

    out_path = os.path.join(script_dir, args.output)
    file_mode = "a" if args.append else "w"
    with open(out_path, file_mode, encoding="utf-8") as log:
        epoch_range = ""
        if args.epoch_min is not None or args.epoch_max is not None:
            epoch_range = f", epoch range: [{args.epoch_min or '*'}-{args.epoch_max or '*'}]"
        log.write(
            f"\nMPC CEM Evaluation Sweep\n"
            f"Checkpoints: {len(checkpoints)}, "
            f"Episodes per checkpoint: {args.episodes}, Seed: {args.seed}{epoch_range}\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        total_t0 = time.time()
        for i, (epoch, ckpt_name) in enumerate(checkpoints):
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            header = (
                f"{'=' * 70}\n"
                f"  [{i + 1}/{len(checkpoints)}] Epoch {epoch} -- {ckpt_name}\n"
                f"{'=' * 70}\n"
            )
            print(header, end="")
            log.write(header)
            log.flush()

            t0 = time.time()
            cmd = [
                sys.executable, mpc_script,
                "--world_model", ckpt_path,
                "--episodes", str(args.episodes),
                "--seed", str(args.seed),
            ]

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

        total_elapsed = time.time() - total_t0
        footer = (
            f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(total: {total_elapsed / 60:.1f} min)\n"
        )
        log.write(footer)
        print(footer)

    print(f"All done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
