# Source logs for paper numbers

Logs and metric sweeps behind numbers in the paper that were not otherwise in the
repository. Run tags follow the paper's run table: `<seed>-<GPU>-<protocol>`, with
C = one continuous process, A = AdamW moments reset at epoch 300, Z = reward head reads z only.

All commands run from the repository root.

| Paper claim | Files | Reproduce |
|---|---|---|
| Reward regression: LunarLander R² 0.24 from (o_t, a_t), 0.98 with o_{t+1} and terminal steps removed; Reacher 0.9998 | `reward_regression/` | `python tools/reward_predictability_test.py --train_dataset lunarlander_train_dataset.npz --val_dataset lunarlander_val_dataset.npz [--filter_terminals --use_next_obs]`; Reacher with `reacher_rssm/reward_predictability_test.py` in the companion reproduction repository |
| Sign table: Spearman of pool-averaged ROF vs MA-7 MPC, 12345-5090-A −0.50, 12345-5090-C −0.26, 12345-5090-Z +0.79, 777-5090-A +0.12 | `metrics/`, `../mpc_eval_logs_{12345-5090-A,12345-5090-C,12345-5090-Z,777-5090-A}.txt` | `python paper/scripts/check_paper_sources.py` (section 1) |
| Late validation loss 34.68 (12345-5090-C) and 32.75 (12345-5090-Z); late good-pool ROF 0.221 and 0.280 | `training/train_worldmodel_logs_12345-5090-{C,Z}.txt`, `metrics/metrics128_*` | `python paper/scripts/check_paper_sources.py` (section 2) |
| MF-AC: 11,835,691 transitions to the epoch-760 checkpoint (65.4× the 180,916-step dataset); ≈18.4M to epoch 1000 | `training/train_modelfree_actorcritic_logs.txt` | `python paper/scripts/check_paper_sources.py` (section 3) |
| A2C across world-model checkpoints (all nine rows) | `a2c_sweeps/` (WM 280 and 315 are in `../`) | `python paper/scripts/check_paper_sources.py` (section 4) |
| CEM-MPC at about 12 Hz on the RTX 5090 | `../eval_heldout/heldout_*_mpc_*.txt` | `python paper/scripts/check_paper_sources.py` (section 5) |
| Estimator noise: 64 vs 128 Jacobian states, r = 0.50, mean difference 0.009, s.d. 0.011, window standard error ≈ 0.0008 | `metrics/metrics_seed12345*_seed12345.txt` (64 states), `metrics/metrics128_seed12345*` (128 states) | `python paper/scripts/rof_64_vs_128.py` |
| Terminal-reward fidelity: crash anticipated at −7 to −10, \|ρ_s\| ≤ 0.38 against MPC drawdown | `terminal_fidelity/*.tsv` (written by `tools/eval_terminal_fidelity.py`) | `python paper/scripts/terminal_fidelity_panel.py --analyze_only --no_plot` |

Saved outputs: `check_paper_sources_output.txt`, `metrics/rof_64_vs_128_output.txt`,
`terminal_fidelity_panel_output.txt`.

Notes:

- `training/train_modelfree_actorcritic_logs.txt` holds two processes: a first run that
  reached epoch 616, and a resume from the epoch-610 checkpoint that logs epochs 611 to 1000.
  The transition count follows the checkpoint's path: epochs 1 to 610 of the first run, then
  the resumed run. Summing every logged epoch line double-counts epochs 611 to 616.
- File names map to run tags as follows: `seed12345continuous` = 12345-5090-C,
  `seed12345resume` = 12345-5090-A, `seed12345zonly` = 12345-5090-Z,
  `seed777resume` = 777-5090-A. `metrics128_*` files use 128 Jacobian states and metric seeds
  12345, 1 and 2; `metrics_*` files use 64 states and metric seed 12345.
- `terminal_fidelity_panel.py` also reads two runs from the companion reproduction
  repository (`../rof-reproduction`: `12345-3080-C` and `12345-3080-Z`).
