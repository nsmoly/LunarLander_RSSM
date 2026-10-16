# Score BC, CEM-MPC and A2C policies on the 100 held-out episodes of the A2C
# re-evaluation (logs/eval_nseeds/), so every policy in Tables 1 and 2 is
# compared on the same episodes.
#
# Each policy is first replayed on its paper selection episodes (seed 12345) as
# a sanity check, then scored on 100 held-out episodes (seed 99999, environment
# seeds 100000-100099). All runs use the 600-step cap. MPC opens a pygame
# window: run it on an interactive desktop. MPC takes about 36 s per episode on
# the 5090 laptop (about 3.5 h for all six policies); BC and A2C take minutes.
#
# Usage (from any folder):
#   powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1
#   powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1 -Policies a2c_wm310_ep1000,mpc_12345-3080-R_wm310
#   powershell -ExecutionPolicy Bypass -File eval_scripts\eval_heldout.ps1 -SkipSanity

param(
    [string[]]$Policies = @(),
    [switch]$SkipSanity
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$BC     = "eval_checkpoints\actor_bc.pt"
$WM280  = "wm_checkpoints\world_model_20260331_085848_epoch_280.pt"
$WM310  = "wm_checkpoints\world_model_20260402_011236_epoch_310.pt"
$WM777A = "eval_checkpoints\world_model_777-5090-A_20260729_035957_epoch_500.pt"
$AC310  = "eval_checkpoints\actor_wm310_20260501_210333_epoch_1000.pt"

# Sanity = number of selection episodes behind the paper's score for that policy.
$table = @(
    @{ Key = "bc_argmax";              Sanity = 20;  Args = @("test_policy.py", "--actor_type", "obs", "--actor", $BC, "--deterministic") },
    @{ Key = "bc_sampled";             Sanity = 20;  Args = @("test_policy.py", "--actor_type", "obs", "--actor", $BC, "--stochastic") },
    @{ Key = "mpc_12345-3080-R_wm280"; Sanity = 20;  Args = @("wm_mpc_policy.py", "--world_model", $WM280) },
    @{ Key = "mpc_12345-3080-R_wm310"; Sanity = 20;  Args = @("wm_mpc_policy.py", "--world_model", $WM310) },
    @{ Key = "mpc_777-5090-A_ep500";   Sanity = 20;  Args = @("wm_mpc_policy.py", "--world_model", $WM777A) },
    @{ Key = "a2c_wm310_ep1000";       Sanity = 100; Args = @("test_policy.py", "--actor_type", "latent", "--actor", $AC310, "--world_model", $WM310, "--deterministic") }
)

# powershell -File passes "a,b" as one string, so split on commas here.
$want = @($Policies | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$selected = if ($want.Count) { @($table | Where-Object { $want -contains $_.Key }) } else { $table }
$unknown = @($want | Where-Object { $_ -notin $table.Key })
if ($unknown.Count) { throw "Unknown policy: $($unknown -join ', '). Known: $($table.Key -join ', ')" }

foreach ($f in @($BC, $WM280, $WM310, $WM777A, $AC310, "test_policy.py", "wm_mpc_policy.py", "eval_scripts\summarize_heldout.py")) {
    if (-not (Test-Path $f)) { throw "Missing: $f" }
}

$OUT = Join-Path (Get-Location) "logs\eval_heldout"
$TS  = Get-Date -Format "yyyyMMdd_HHmmss"
$CAP = 600
New-Item -ItemType Directory -Force $OUT | Out-Null

$runs = @()
if (-not $SkipSanity) { $runs += $selected | ForEach-Object { @{ P = $_; Seed = 12345; Episodes = $_.Sanity } } }
$runs += $selected | ForEach-Object { @{ P = $_; Seed = 99999; Episodes = 100 } }

$env:PYTHONIOENCODING = "utf-8"
$started = Get-Date
foreach ($r in $runs) {
    $name = "$($r.P.Key)_seed$($r.Seed)_n$($r.Episodes)"
    $log = Join-Path $OUT "heldout_${TS}_$name.txt"
    $pyArgs = @("-u") + $r.P.Args + @("--episodes", $r.Episodes, "--seed", $r.Seed, "--max_steps", $CAP)
    Write-Host "`n=== $name  ->  $log ===" -ForegroundColor Cyan
    [System.IO.File]::WriteAllText($log, "# python $($pyArgs -join ' ')`r`n")
    $t0 = Get-Date
    # Continue: Windows PowerShell 5.1 turns python's stderr lines into terminating errors under Stop.
    $ErrorActionPreference = "Continue"
    & python @pyArgs 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        [System.IO.File]::AppendAllText($log, $line + "`r`n")
    }
    $exit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $mean = Select-String -Path $log -Pattern '^\[Run\] mean_return=([+-]?\d+\.\d+)' | Select-Object -First 1
    $meanText = if ($mean) { $mean.Matches[0].Groups[1].Value } else { "n/a" }
    Write-Host ("{0}: exit={1} mean_return={2} ({3:N1} min)" -f $name, $exit, $meanText, ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
    if ($exit -ne 0) { Write-Host "WARNING: $name exited with code $exit" -ForegroundColor Yellow }
}

Write-Host ("`nAll runs done in {0:N1} min. Summary (latest log of each policy):" -f ((Get-Date) - $started).TotalMinutes) -ForegroundColor Green
& python eval_scripts\summarize_heldout.py
