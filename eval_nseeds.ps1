# Re-evaluate the selected A2C policies on fresh episodes (J's review item #5).
#
#   WM-AC : actor epoch 800 trained on world-model checkpoint 280 (12345-3080-R)
#   MF-AC : model-free actor epoch 760
#
# Step 0 replays the original selection protocol (seed 12345) and must reproduce
# +217.48 (WM-AC, 600-step cap) and +193.04 (MF-AC, 1000-step cap).
# Step 1 scores the same frozen policies on 100 unused episodes (seed 99999,
# episode seeds 100000-100099) with the paper's caps.
# Step 2 adds WM-AC at the common 1000-step cap, for the footnote only.
#
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File eval_nseeds.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$A    = "archive\V4_goodBest_landing_extendedTrainingBest_04022026"
$WM   = "$A\checkpoints_worldmodel\world_model_20260331_085848_epoch_280.pt"
$WMAC = "$A\checkpoints_actorcritic_wms\checkpoints_actorcritic_wm280\actor_20260503_031907_epoch_800.pt"
$MFAC = "$A\checkpoints_actorcritic_modelfree\actor_mf_20260324_231147_epoch_760.pt"
$OUT  = "logs\eval_nseeds"
$TS   = Get-Date -Format "yyyyMMdd_HHmmss"
$EPISODES = 100
$FRESH_SEED = 99999

foreach ($f in @($WM, $WMAC, $MFAC)) {
    if (-not (Test-Path $f)) { throw "Missing checkpoint: $f" }
}
New-Item -ItemType Directory -Force $OUT | Out-Null

$runs = @(
    @{ Name = "wmac800_seed12345_cap600";  Type = "latent"; Actor = $WMAC; Seed = 12345;       Cap = 600;  Expect = "+217.48" },
    @{ Name = "mfac760_seed12345_cap1000"; Type = "obs";    Actor = $MFAC; Seed = 12345;       Cap = 1000; Expect = "+193.04" },
    @{ Name = "wmac800_seed${FRESH_SEED}_cap600";  Type = "latent"; Actor = $WMAC; Seed = $FRESH_SEED; Cap = 600;  Expect = $null },
    @{ Name = "mfac760_seed${FRESH_SEED}_cap1000"; Type = "obs";    Actor = $MFAC; Seed = $FRESH_SEED; Cap = 1000; Expect = $null },
    @{ Name = "wmac800_seed${FRESH_SEED}_cap1000"; Type = "latent"; Actor = $WMAC; Seed = $FRESH_SEED; Cap = 1000; Expect = $null }
)

$summary = @("Re-evaluation session $TS  (episodes=$EPISODES, deterministic)", "")
$started = Get-Date

foreach ($r in $runs) {
    $log = "$OUT\reeval_${TS}_$($r.Name).txt"
    Write-Host "`n=== $($r.Name)  ->  $log ===" -ForegroundColor Cyan

    $pyArgs = @("-u", "test_policy.py", "--actor_type", $r.Type, "--actor", $r.Actor,
                "--episodes", $EPISODES, "--seed", $r.Seed, "--max_steps", $r.Cap, "--deterministic")
    if ($r.Type -eq "latent") { $pyArgs += @("--world_model", $WM) }

    $ErrorActionPreference = "Continue"
    & python @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $exit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    $scoreLines = Select-String -Path $log -Pattern "^\[Run\] (mean_return|perfect)" | ForEach-Object { $_.Line }
    $mean = if ($scoreLines -and ($scoreLines[0] -match "mean_return=([+-]?\d+\.\d+)")) { $Matches[1] } else { "n/a" }

    $status = "exit=$exit"
    if ($r.Expect) {
        $status += if ($mean -eq $r.Expect) { "  REPRODUCED ($mean)" } else { "  MISMATCH: got $mean, expected $($r.Expect)" }
    }
    $summary += "[$($r.Name)] $status"
    $summary += $scoreLines
    $summary += ""
}

$summary += "Total time: {0:N1} min" -f ((Get-Date) - $started).TotalMinutes
$summaryFile = "$OUT\reeval_${TS}_SUMMARY.txt"
$summary | Set-Content -Encoding utf8 $summaryFile

Write-Host "`n===== SUMMARY ($summaryFile) =====" -ForegroundColor Green
$summary | ForEach-Object { Write-Host $_ }
