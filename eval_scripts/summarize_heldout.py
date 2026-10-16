"""Summarise the held-out runs in Table 2's columns, next to the held-out A2C runs of Table 1.

    python eval_scripts/summarize_heldout.py
    python eval_scripts/summarize_heldout.py --ts 20261014_143234 20261014_223629

Reads the logs eval_heldout.ps1 writes to logs/eval_heldout/
(heldout_<session>_<policy>_seed<seed>_n<episodes>.txt), taking the latest
session of each policy, or with --ts the first listed session that has it. Also
reads the A2C re-evaluation logs in logs/eval_nseeds/ (UTF-16; encoding is
detected per file). Episode k of every run uses environment seed seed+k, so
runs with the same seed are paired by k. Writes logs/eval_heldout/SUMMARY.txt.
"""

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "logs" / "eval_heldout"
WMAC = "WM-AC (WM 280, A2C ep 800)"
MFAC = "MF-AC (ep 760), 1000-step cap"
A2C = {
    WMAC: ROOT / "logs" / "eval_nseeds" / "reeval_20260928_235825_wmac800_seed99999_cap600.txt",
    MFAC: ROOT / "logs" / "eval_nseeds" / "reeval_20260928_235825_mfac760_seed99999_cap1000.txt",
}
# key, label, sanity episodes, value on the selection episodes (seed 12345) in the paper, decimals quoted
RUNS = [
    ("bc_argmax", "BC, deterministic (argmax)", 20, -190.2, 1),
    ("bc_sampled", "BC, stochastic (sampled)", 20, 185.6, 1),
    ("mpc_12345-3080-R_wm280", "MPC, 12345-3080-R, WM 280", 20, 162.83, 2),
    ("mpc_12345-3080-R_wm310", "MPC, 12345-3080-R, WM 310", 20, 156.0, 1),
    ("mpc_777-5090-A_ep500", "MPC, 777-5090-A, epoch 500", 20, 216.56, 2),
    ("a2c_wm310_ep1000", "A2C on WM 310 (A2C ep 1000)", 100, 209.60, 2),
]
LABEL = {key: label for key, label, *_ in RUNS}
EPISODE_RE = re.compile(r"^\[Episode (\d+)\] return=([+-]?\d+\.\d+), steps=(\d+)")
SESSION_RE = re.compile(r"^heldout_(\d{8}_\d{6})_")


def read_text(path):
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def episodes(path):
    out = {}
    for line in read_text(path).splitlines():
        m = EPISODE_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
    return out


def find(ts_list, key, seed, n):
    suffix = f"_{key}_seed{seed}_n{n}.txt"
    if ts_list:
        hits = [HELDOUT / f"heldout_{ts}{suffix}" for ts in ts_list]
        return next((f for f in hits if f.exists()), None)
    hits = sorted(f for f in HELDOUT.glob(f"heldout_*{suffix}")
                  if (m := SESSION_RE.match(f.name)) and f.name == f"heldout_{m.group(1)}{suffix}")
    return hits[-1] if hits else None


def row(name, eps):
    r = np.array([v[0] for v in eps.values()])
    s = np.array([v[1] for v in eps.values()])
    n = len(r)
    return (f"{name:<34s} {r.mean():+8.2f} {r.min():+8.2f} {int((r > 200).sum()):>4d}/{n:<3d} "
            f"{int((r < 0).sum()):>5d} {int((r < -100).sum()):>6d} {s.mean():>7.1f}")


def paired_ci(a, b, n_boot=10000, seed=0):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k][0] - b[k][0] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ts", nargs="*", default=[], help="session(s) to read, in order of preference (default: latest per policy)")
    p.add_argument("--out", default=None, help="summary path (default: logs/eval_heldout/SUMMARY.txt)")
    a = p.parse_args()

    header = (f"{'Policy':<34s} {'Mean':>8s} {'Worst':>8s} {'Perfect':>8s} "
              f"{'Neg.':>5s} {'Catast.':>6s} {'Steps':>7s}")
    sessions = set()

    def load(key, seed, n):
        f = find(a.ts, key, seed, n)
        if not f:
            return {}
        sessions.add(SESSION_RE.match(f.name).group(1))
        return episodes(f)

    lines = ["== Sanity check: selection episodes (seed 12345), expected = value in the paper ==",
             "   (12345-3080-R MPC was swept on the RTX 3080; on the 5090 a mismatch is expected)", header]
    for key, label, n, want, nd in RUNS:
        eps = load(key, 12345, n)
        if not eps:
            lines.append(f"{label:<34s} (not run)")
            continue
        mean = np.mean([v[0] for v in eps.values()])
        verdict = "REPRODUCED" if round(mean, nd) == want else f"MISMATCH (paper: {want:+.{nd}f})"
        lines.append(row(label, eps) + f"   {verdict}")
    lines.append("")

    held = {}
    lines += ["== Held-out: 100 episodes (seed 99999, env seeds 100000-100099) ==", header]
    for key, label, *_ in RUNS:
        eps = load(key, 99999, 100)
        if eps:
            held[label] = eps
            lines.append(row(label, eps))
        else:
            lines.append(f"{label:<34s} (not run)")
    for label, f in A2C.items():
        if f.exists():
            held[label] = episodes(f)
            lines.append(row(label, held[label]))
    for label, eps in held.items():
        if len(eps) != 100:
            lines.append(f"WARNING: {label} has {len(eps)} episodes, expected 100")
    lines.append("")

    lines += ["== Paired differences on the held-out episodes (A - B, bootstrap 95% CI, 10,000 resamples) =="]
    pairs = [
        (LABEL["bc_sampled"], LABEL["mpc_12345-3080-R_wm280"]),
        (LABEL["bc_sampled"], LABEL["mpc_12345-3080-R_wm310"]),
        (LABEL["bc_sampled"], LABEL["mpc_777-5090-A_ep500"]),
        (LABEL["bc_sampled"], WMAC),
        (WMAC, LABEL["mpc_12345-3080-R_wm280"]),
        (LABEL["a2c_wm310_ep1000"], LABEL["mpc_12345-3080-R_wm310"]),
        (WMAC, LABEL["a2c_wm310_ep1000"]),
        (LABEL["mpc_12345-3080-R_wm310"], LABEL["mpc_12345-3080-R_wm280"]),
        (LABEL["mpc_777-5090-A_ep500"], LABEL["mpc_12345-3080-R_wm280"]),
        (LABEL["mpc_777-5090-A_ep500"], LABEL["mpc_12345-3080-R_wm310"]),
        (WMAC, MFAC),
    ]
    for x, y in pairs:
        if x in held and y in held:
            m, lo, hi, n = paired_ci(held[x], held[y])
            lines.append(f"{x}  minus  {y}: {m:+.1f}  [{lo:+.1f}, {hi:+.1f}]  (n={n})")
    lines += ["", "The last line is the Table 1 comparison (different caps), included as a check against the paper."]

    lines = [f"Held-out evaluation, sessions {', '.join(sorted(sessions)) or '(none)'}",
             "Perfect: return > +200; Neg.: < 0; Catast.: < -100. 600-step cap unless noted.", ""] + lines
    text = "\n".join(lines)
    print(text)
    out = Path(a.out) if a.out else HELDOUT / "SUMMARY.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
