# Held-out evaluation of BC, CEM-MPC and A2C

BC and CEM-MPC scored on the 100 held-out episodes used for the A2C
re-evaluation in `logs/eval_nseeds/`, so that every policy in Tables 1 and 2 is
compared on the same episodes.

## Protocol

- Episode `k` (k = 1..N) uses environment seed `seed + k`. Held-out: seed 99999,
  100 episodes (environment seeds 100000-100099). Selection episodes (as in the
  paper): seed 12345.
- 600-step cap for every run here. MF-AC (from `logs/eval_nseeds/`) uses the
  standard 1000-step cap.
- Each policy is first replayed on its paper selection episodes (seed 12345,
  same episode count as the paper) as a sanity check, then scored on the
  held-out episodes.
- All runs on an RTX 5090 Laptop GPU, Python 3.11.9, torch 2.11.0+cu128.

## Policies and checkpoints

| Policy | Checkpoint(s) | SHA-256 (first 16) |
|---|---|---|
| BC, argmax and sampled | `eval_checkpoints/actor_bc.pt` (validation-best, epoch 9) | `159af2dc0c365164` |
| MPC, `12345-3080-R` WM 280 | `wm_checkpoints/world_model_20260331_085848_epoch_280.pt` | `723f921ffb07071f` |
| MPC, `12345-3080-R` WM 310 | `wm_checkpoints/world_model_20260402_011236_epoch_310.pt` | `b0edcb22637a60d9` |
| MPC, `777-5090-A` epoch 500 | `eval_checkpoints/world_model_777-5090-A_20260729_035957_epoch_500.pt` | `03e2dec512030479` |
| A2C on WM 310, A2C epoch 1000 | `eval_checkpoints/actor_wm310_20260501_210333_epoch_1000.pt` + WM 310 | `d9be663cd4f1cf96` |
| WM-AC on WM 280, A2C epoch 800 (`logs/eval_nseeds/`) | `actor.pt` + WM 280 | `28b8aa05f2d27761` |
| MF-AC, epoch 760 (`logs/eval_nseeds/`) | `actor_mf.pt` | `63895f8911a279f5` |

The `# python` line at the top of each log shows the local paths the runs were
made from; the repository copies are byte-identical (same SHA-256).

## Files

- `heldout_<session>_<policy>_seed<seed>_n<episodes>.txt`: per-episode logs
  from `eval_scripts/eval_heldout.ps1`. Session `20261014_143234` has BC and MPC
  on WM 280 and `777-5090-A`; session `20261014_223629` has MPC and A2C on WM 310.
- `SUMMARY.txt`: all policies together (sanity checks, held-out table, paired
  bootstrap differences), from `eval_scripts/summarize_heldout.py`.

## Results (100 held-out episodes)

| Policy | Mean | Worst | Perfect (> +200) | Catastrophic (< -100) |
|---|---:|---:|---:|---:|
| A2C on WM 310 (ep 1000) | +192.33 | -181.52 | 68 | 1 |
| WM-AC (WM 280, ep 800) | +189.53 | -419.91 | 70 | 5 |
| MF-AC (ep 760), 1000-step cap | +183.67 | -153.11 | 49 | 1 |
| MPC, `777-5090-A` ep 500 | +148.64 | -217.09 | 57 | 3 |
| MPC, WM 310 | +145.85 | -162.16 | 57 | 5 |
| MPC, WM 280 | +143.97 | -180.34 | 58 | 5 |
| BC, sampled | +116.60 | -281.63 | 51 | 9 |
| BC, argmax | -47.79 | -435.86 | 25 | 46 |

Paired differences on the same episodes (bootstrap 95% interval, 10,000
resamples, rng seed 0): A2C minus MPC on the same world model is +46.5
[+18.4, +75.4] on WM 310 and +45.6 [+14.5, +76.1] on WM 280; WM-AC minus A2C on
WM 310 is -2.8 [-33.2, +27.3]; MPC on WM 310 minus MPC on WM 280 is +1.9
[-19.9, +24.0]; BC sampled minus MPC is -27.4 to -32.0 with every interval
including zero. The full list is in `SUMMARY.txt`.

Sanity checks: BC (both modes), MPC on `777-5090-A` and A2C on WM 310 reproduce
their paper selection scores exactly. MPC on WM 280 and WM 310 does not (+168.03
and +135.23 against +162.8 and +156.0): those sweeps were run on the RTX 3080
with torch 1.12, and CEM-MPC returns do not replay across the two machines.

## Reproduce

On an interactive desktop (MPC opens a pygame window):

```
powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1
powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1 -Policies a2c_wm310_ep1000,mpc_12345-3080-R_wm310
powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1 -SkipSanity
```

Policies: `bc_argmax`, `bc_sampled`, `mpc_12345-3080-R_wm280`,
`mpc_12345-3080-R_wm310`, `mpc_777-5090-A_ep500`, `a2c_wm310_ep1000` (default:
all). `-SkipSanity` skips the seed-12345 replays. All six take about 3.5 h on
the 5090, almost all of it MPC. The script ends by running
`python eval_scripts/summarize_heldout.py`, which summarises the latest log of
each policy into `SUMMARY.txt` (`--ts <session> ...` pins specific sessions).
