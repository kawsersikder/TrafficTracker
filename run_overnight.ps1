<#
  TrafficTracker - overnight experiment orchestrator
  Resumable: every job writes a sentinel into .run_state\ when it succeeds.
  Re-running this script skips completed jobs, so a power cut costs at most
  the single job that was in flight (YOLO jobs resume mid-training).
  ASCII ONLY - PowerShell 5.1 reads this file as ANSI.
#>

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Repo

$PY       = Join-Path $Repo ".venv\Scripts\python.exe"
$YOLO     = Join-Path $Repo ".venv\Scripts\yolo.exe"
$State    = Join-Path $Repo ".run_state"
$Logs     = Join-Path $Repo "logs"
$Progress = Join-Path $Repo "RUN_PROGRESS.md"

New-Item -ItemType Directory -Force -Path $State, $Logs, "F:\temp" | Out-Null

if (-not (Test-Path $PY))   { Write-Host "FATAL: no python at $PY" -ForegroundColor Red; pause; exit 1 }
if (-not (Test-Path $YOLO)) { Write-Host "WARN: no yolo.exe at $YOLO - vision jobs will fail" -ForegroundColor Yellow }

# Keep everything off the C: drive (C: filled up mid-session before).
$env:TEMP = "F:\temp"; $env:TMP = "F:\temp"
$env:CURL_CA_BUNDLE = ""
$env:PYTHONUNBUFFERED = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Keep the machine awake without changing power settings.
$sig = '[DllImport("kernel32.dll", SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags);'
try {
    Add-Type -Name Power -Namespace Win32 -MemberDefinition $sig -ErrorAction Stop
    [void][Win32.Power]::SetThreadExecutionState(2147483648 -bor 1 -bor 64)
    Write-Host "sleep inhibited for the duration of the run"
} catch { Write-Host "could not inhibit sleep: $_" -ForegroundColor Yellow }

$script:Jobs      = [System.Collections.ArrayList]::new()
$script:Results   = [ordered]@{}
$script:StartedAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

function Add-Job([string]$Name, [string]$Desc, [scriptblock]$Body, [string]$Marker) {
    [void]$script:Jobs.Add([pscustomobject]@{ Name=$Name; Desc=$Desc; Body=$Body; Marker=$Marker })
}

function Write-Ledger([string]$Current) {
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("# Overnight run - live progress")
    [void]$L.Add("")
    [void]$L.Add("Started: $script:StartedAt")
    [void]$L.Add("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("Host: RTX 2060 6GB / Ryzen 5 2600 / 24GB RAM")
    [void]$L.Add("")
    [void]$L.Add("Resume after a power cut: run RUN_OVERNIGHT.bat again.")
    [void]$L.Add("Finished jobs are skipped; YOLO jobs resume from their last epoch.")
    [void]$L.Add("")
    $done=0; $fail=0; $pend=0
    foreach ($j in $script:Jobs) {
        $r = $script:Results[$j.Name]
        if ($r -and $r.Status -eq "done") { $done++ }
        elseif ($r -and $r.Status -like "FAILED*") { $fail++ }
        elseif (Test-Path (Join-Path $State ($j.Name + ".done"))) { $done++ }
        else { $pend++ }
    }
    [void]$L.Add("Summary: $done done / $fail failed / $pend remaining (of $($script:Jobs.Count))")
    [void]$L.Add("")
    [void]$L.Add("| # | job | status | started | duration |")
    [void]$L.Add("|---|-----|--------|---------|----------|")
    $i = 0
    foreach ($j in $script:Jobs) {
        $i++
        $r = $script:Results[$j.Name]
        if ($r) { $st=$r.Status; $stt=$r.Started; $dur=$r.Duration }
        elseif (Test-Path (Join-Path $State ($j.Name + ".done"))) { $st="done (earlier run)"; $stt="-"; $dur="-" }
        elseif ($j.Name -eq $Current) { $st="RUNNING"; $stt=(Get-Date -Format "HH:mm:ss"); $dur="..." }
        else { $st="pending"; $stt="-"; $dur="-" }
        [void]$L.Add("| $i | " + $j.Desc + " | " + $st + " | " + $stt + " | " + $dur + " |")
    }
    [void]$L.Add("")
    [void]$L.Add("Logs: logs\<job>.log   Sentinels: .run_state\")
    Set-Content -Path $Progress -Value ($L -join "`r`n") -Encoding ASCII
}

function Invoke-Job($Job) {
    $sentinel = Join-Path $State ($Job.Name + ".done")
    $log = Join-Path $Logs ($Job.Name + ".log")

    # A previous run may have finished this job without leaving a sentinel
    # (older builds mis-read tqdm stderr as failure). Trust the log instead.
    if (-not (Test-Path $sentinel) -and (Test-Path $log)) {
        $prev = (Get-Content -Path $log -Raw -ErrorAction SilentlyContinue)
        if ($null -eq $prev) { $prev = "" }
        if (($prev -notmatch "Traceback \(most recent call last\)") -and
            $Job.Marker -and ($prev -match $Job.Marker)) {
            New-Item -ItemType File -Force -Path $sentinel | Out-Null
        }
    }

    if (Test-Path $sentinel) {
        Write-Host ("[skip] " + $Job.Name) -ForegroundColor DarkGray
        return
    }
    $t0  = Get-Date
    Write-Host ""
    Write-Host ("[run ] " + $Job.Name + " - " + $Job.Desc) -ForegroundColor Cyan
    $script:Results[$Job.Name] = [pscustomobject]@{ Status="RUNNING"; Started=$t0.ToString("HH:mm:ss"); Duration="..." }
    Write-Ledger $Job.Name

    $global:LASTEXITCODE = 0
    try {
        & $Job.Body 2>&1 | Tee-Object -FilePath $log
    } catch {
        ($_ | Out-String) | Add-Content -Path $log
    }

    $txt = ""
    if (Test-Path $log) { $txt = (Get-Content -Path $log -Raw -ErrorAction SilentlyContinue) }
    if ($null -eq $txt) { $txt = "" }
    $ok = $true
    if ($txt -match "Traceback \(most recent call last\)") { $ok = $false }
    elseif ($Job.Marker -and ($txt -notmatch $Job.Marker)) { $ok = $false }

    $dur = "{0:hh\:mm\:ss}" -f (New-TimeSpan -Start $t0 -End (Get-Date))
    if ($ok) {
        New-Item -ItemType File -Force -Path $sentinel | Out-Null
        $script:Results[$Job.Name] = [pscustomobject]@{ Status="done"; Started=$t0.ToString("HH:mm:ss"); Duration=$dur }
        Write-Host ("[ok  ] " + $Job.Name + " in " + $dur) -ForegroundColor Green
    } else {
        $script:Results[$Job.Name] = [pscustomobject]@{ Status="FAILED (see log)"; Started=$t0.ToString("HH:mm:ss"); Duration=$dur }
        Write-Host ("[FAIL] " + $Job.Name + " after " + $dur + " - continuing") -ForegroundColor Red
    }
    Write-Ledger $null
}

$MANILA = "experiments\out\manila_segments.parquet"
$LOOP   = "experiments\out\sathorn.parquet"
$CCTV   = "experiments\out\sathorn_cctv.parquet"

Add-Job "00_preflight" "Preflight: venv, torch, CUDA, input files" -Marker "PREFLIGHT OK" { & $PY experiments\_preflight.py }

Add-Job "01_loop_baselines" "Sathorn loop-coil baselines (h1/h2/h4)" -Marker "all baselines done" {
    foreach ($h in 1,2,4) { & $PY experiments\baselines.py --data $LOOP --horizon $h }
}
Add-Job "02_loop_gru_h1" "Sathorn loop-coil GRU h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model gru --horizon 1
}
Add-Job "02_loop_gru_h2" "Sathorn loop-coil GRU h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model gru --horizon 2
}
Add-Job "02_loop_gru_h4" "Sathorn loop-coil GRU h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model gru --horizon 4
}
Add-Job "02_loop_lstm_h1" "Sathorn loop-coil LSTM h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model lstm --horizon 1
}
Add-Job "02_loop_lstm_h2" "Sathorn loop-coil LSTM h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model lstm --horizon 2
}
Add-Job "02_loop_lstm_h4" "Sathorn loop-coil LSTM h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model lstm --horizon 4
}
Add-Job "02_loop_tcn_h1" "Sathorn loop-coil TCN h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model tcn --horizon 1
}
Add-Job "02_loop_tcn_h2" "Sathorn loop-coil TCN h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model tcn --horizon 2
}
Add-Job "02_loop_tcn_h4" "Sathorn loop-coil TCN h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $LOOP --model tcn --horizon 4
}
Add-Job "03_cctv_prep" "Build Sathorn CCTV parquet (4356 files, 2016-2019)" -Marker "wrote .*parquet" {
    & $PY experiments\sathorn\prepare_sathorn.py --csv "dataset/bangkok_sathorn-intersection/extracted/cctv/cctv-camera/Link*/*_volume_*.csv" --time-col Time --date-from-filename --series-from-parent --value-col "E1,E2,E3,S1,S2,S3,W1,W2,W3,W4,N1,N2,N3" --mode volume --resample 15min --per-file-resample --out $CCTV
}

Add-Job "04_cctv_baselines" "Sathorn CCTV baselines (h1/h2/h4)" -Marker "all baselines done" {
    foreach ($h in 1,2,4) { & $PY experiments\baselines.py --data $CCTV --horizon $h }
}
Add-Job "05_cctv_gru_h1" "Sathorn CCTV GRU h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model gru --horizon 1
}
Add-Job "05_cctv_gru_h2" "Sathorn CCTV GRU h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model gru --horizon 2
}
Add-Job "05_cctv_gru_h4" "Sathorn CCTV GRU h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model gru --horizon 4
}
Add-Job "05_cctv_lstm_h1" "Sathorn CCTV LSTM h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model lstm --horizon 1
}
Add-Job "05_cctv_lstm_h2" "Sathorn CCTV LSTM h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model lstm --horizon 2
}
Add-Job "05_cctv_lstm_h4" "Sathorn CCTV LSTM h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model lstm --horizon 4
}
Add-Job "05_cctv_tcn_h1" "Sathorn CCTV TCN h1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model tcn --horizon 1
}
Add-Job "05_cctv_tcn_h2" "Sathorn CCTV TCN h2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model tcn --horizon 2
}
Add-Job "05_cctv_tcn_h4" "Sathorn CCTV TCN h4" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $CCTV --model tcn --horizon 4
}
Add-Job "06_transfer_cctv" "Transfer Manila -> Bangkok CCTV (h1)" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $CCTV --pretrained experiments\runs\manila_segments_gru_cls_h1\best.pt --k-days 1 3 7 14 28
}
Add-Job "07_transfer_loop_h2" "Transfer Manila -> Sathorn loop-coil h2" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $LOOP --pretrained experiments\runs\manila_segments_gru_cls_h2\best.pt --horizon 2 --k-days 1 3 7 14 28
}

Add-Job "08_transfer_edsa_h2" "Transfer within-Manila -> EDSA h2" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $MANILA --target-prefix "EDSA|" --horizon 2 --k-days 1 3 7 14 28
}
Add-Job "07_transfer_loop_h4" "Transfer Manila -> Sathorn loop-coil h4" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $LOOP --pretrained experiments\runs\manila_segments_gru_cls_h4\best.pt --horizon 4 --k-days 1 3 7 14 28
}

Add-Job "08_transfer_edsa_h4" "Transfer within-Manila -> EDSA h4" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $MANILA --target-prefix "EDSA|" --horizon 4 --k-days 1 3 7 14 28
}
Add-Job "09_seed1_gru_h1" "Manila GRU h1 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 1 --seed 1 --out "experiments\runs\seed1_manila_segments_gru_cls_h1"
}
Add-Job "09_seed1_gru_h2" "Manila GRU h2 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 2 --seed 1 --out "experiments\runs\seed1_manila_segments_gru_cls_h2"
}
Add-Job "09_seed1_gru_h4" "Manila GRU h4 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 4 --seed 1 --out "experiments\runs\seed1_manila_segments_gru_cls_h4"
}
Add-Job "09_seed1_lstm_h1" "Manila LSTM h1 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 1 --seed 1 --out "experiments\runs\seed1_manila_segments_lstm_cls_h1"
}
Add-Job "09_seed1_lstm_h2" "Manila LSTM h2 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 2 --seed 1 --out "experiments\runs\seed1_manila_segments_lstm_cls_h2"
}
Add-Job "09_seed1_lstm_h4" "Manila LSTM h4 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 4 --seed 1 --out "experiments\runs\seed1_manila_segments_lstm_cls_h4"
}
Add-Job "09_seed1_tcn_h1" "Manila TCN h1 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 1 --seed 1 --out "experiments\runs\seed1_manila_segments_tcn_cls_h1"
}
Add-Job "09_seed1_tcn_h2" "Manila TCN h2 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 2 --seed 1 --out "experiments\runs\seed1_manila_segments_tcn_cls_h2"
}
Add-Job "09_seed1_tcn_h4" "Manila TCN h4 seed 1" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 4 --seed 1 --out "experiments\runs\seed1_manila_segments_tcn_cls_h4"
}
Add-Job "09_seed2_gru_h1" "Manila GRU h1 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 1 --seed 2 --out "experiments\runs\seed2_manila_segments_gru_cls_h1"
}
Add-Job "09_seed2_gru_h2" "Manila GRU h2 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 2 --seed 2 --out "experiments\runs\seed2_manila_segments_gru_cls_h2"
}
Add-Job "09_seed2_gru_h4" "Manila GRU h4 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model gru --horizon 4 --seed 2 --out "experiments\runs\seed2_manila_segments_gru_cls_h4"
}
Add-Job "09_seed2_lstm_h1" "Manila LSTM h1 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 1 --seed 2 --out "experiments\runs\seed2_manila_segments_lstm_cls_h1"
}
Add-Job "09_seed2_lstm_h2" "Manila LSTM h2 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 2 --seed 2 --out "experiments\runs\seed2_manila_segments_lstm_cls_h2"
}
Add-Job "09_seed2_lstm_h4" "Manila LSTM h4 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model lstm --horizon 4 --seed 2 --out "experiments\runs\seed2_manila_segments_lstm_cls_h4"
}
Add-Job "09_seed2_tcn_h1" "Manila TCN h1 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 1 --seed 2 --out "experiments\runs\seed2_manila_segments_tcn_cls_h1"
}
Add-Job "09_seed2_tcn_h2" "Manila TCN h2 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 2 --seed 2 --out "experiments\runs\seed2_manila_segments_tcn_cls_h2"
}
Add-Job "09_seed2_tcn_h4" "Manila TCN h4 seed 2" -Marker "checkpoint: " {
    & $PY experiments\train_forecaster.py --data $MANILA --model tcn --horizon 4 --seed 2 --out "experiments\runs\seed2_manila_segments_tcn_cls_h4"
}
Add-Job "10_aggregate_mid" "Regenerate tables + figures (time-series)" -Marker "results_table|wrote" {
    & $PY experiments\analysis\aggregate_results.py
}

Add-Job "11_yolov8s_finish" "Finish YOLOv8s (epochs 44-50)" -Marker "Results saved to" {
    & $YOLO detect train resume model="runs\detect\runs\detect\dhakaai_yolov8s\weights\last.pt" workers=0
}

Add-Job "12_yolov8m" "YOLOv8m 40 epochs on DhakaAI (approx 4h)" -Marker "Results saved to" {
    $last = "runs\detect\dhakaai_yolov8m\weights\last.pt"
    if (Test-Path $last) {
        & $YOLO detect train resume model=$last workers=0
    } else {
        & $YOLO detect train model=yolov8m.pt data="experiments\out\dhakaai_yolo\dataset.yaml" epochs=40 imgsz=640 batch=8 workers=0 device=0 amp=True cache=False patience=15 project="runs\detect" name="dhakaai_yolov8m" exist_ok=True
    }
}

Add-Job "13_aggregate_final" "Final aggregate + artifact inventory" -Marker "results_table|wrote" {
    & $PY experiments\analysis\aggregate_results.py
    & $PY experiments\_inventory.py
}

Write-Ledger $null
Write-Host ""
Write-Host ("TrafficTracker overnight run - " + $script:Jobs.Count + " jobs") -ForegroundColor Yellow
foreach ($j in $script:Jobs) { Invoke-Job $j }

Write-Ledger $null
Write-Host ""
Write-Host "ALL JOBS ATTEMPTED. See RUN_PROGRESS.md" -ForegroundColor Yellow
[void][Win32.Power]::SetThreadExecutionState(2147483648)
