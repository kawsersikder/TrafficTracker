<#
  TrafficTracker - phase 3: Dhaka cross-dataset generalisation (TFP-BD vs DhakaAI).
  Same resumable design as run_overnight.ps1 (sentinels in .run_state3\).
  ASCII ONLY - PowerShell 5.1 reads this file as ANSI.
#>

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Repo

$PY       = Join-Path $Repo ".venv\Scripts\python.exe"
$State    = Join-Path $Repo ".run_state3"
$Logs     = Join-Path $Repo "logs3"
$Progress = Join-Path $Repo "RUN_PROGRESS_PHASE3.md"

New-Item -ItemType Directory -Force -Path $State, $Logs, "F:\temp" | Out-Null
if (-not (Test-Path $PY)) { Write-Host "FATAL: no python at $PY" -ForegroundColor Red; pause; exit 1 }

$env:TEMP = "F:\temp"; $env:TMP = "F:\temp"
$env:PYTHONUNBUFFERED = "1"; $env:KMP_DUPLICATE_LIB_OK = "TRUE"

$sig = '[DllImport("kernel32.dll", SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags);'
try {
    Add-Type -Name Power3 -Namespace Win32 -MemberDefinition $sig -ErrorAction Stop
    [void][Win32.Power3]::SetThreadExecutionState(2147483648 -bor 1 -bor 64)
} catch { Write-Host "could not inhibit sleep: $_" -ForegroundColor Yellow }

$script:Jobs      = [System.Collections.ArrayList]::new()
$script:Results   = [ordered]@{}
$script:StartedAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

function Add-Job([string]$Name, [string]$Desc, [scriptblock]$Body, [string]$Marker) {
    [void]$script:Jobs.Add([pscustomobject]@{ Name=$Name; Desc=$Desc; Body=$Body; Marker=$Marker })
}

function Write-Ledger([string]$Current) {
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("# Phase 3 run - live progress")
    [void]$L.Add("")
    [void]$L.Add("Started: $script:StartedAt")
    [void]$L.Add("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("")
    [void]$L.Add("Purpose: train on TFP-BD (24k Dhaka frames, never used for training)")
    [void]$L.Add("and cross-evaluate against DhakaAI on a shared class vocabulary.")
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
    Set-Content -Path $Progress -Value ($L -join "`r`n") -Encoding ASCII
}

function Invoke-Job($Job) {
    $sentinel = Join-Path $State ($Job.Name + ".done")
    $log = Join-Path $Logs ($Job.Name + ".log")
    if (-not (Test-Path $sentinel) -and (Test-Path $log)) {
        $prev = (Get-Content -Path $log -Raw -ErrorAction SilentlyContinue)
        if ($null -eq $prev) { $prev = "" }
        if (($prev -notmatch "Traceback \(most recent call last\)") -and
            $Job.Marker -and ($prev -match $Job.Marker)) {
            New-Item -ItemType File -Force -Path $sentinel | Out-Null
        }
    }
    if (Test-Path $sentinel) { Write-Host ("[skip] " + $Job.Name) -ForegroundColor DarkGray; return }

    $t0 = Get-Date
    Write-Host ""
    Write-Host ("[run ] " + $Job.Name + " - " + $Job.Desc) -ForegroundColor Cyan
    $script:Results[$Job.Name] = [pscustomobject]@{ Status="RUNNING"; Started=$t0.ToString("HH:mm:ss"); Duration="..." }
    Write-Ledger $Job.Name

    try { & $Job.Body 2>&1 | Tee-Object -FilePath $log }
    catch { ($_ | Out-String) | Add-Content -Path $log }

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
        Write-Host ("[FAIL] " + $Job.Name + " after " + $dur) -ForegroundColor Red
    }
    Write-Ledger $null
}

$YOLO   = Join-Path $Repo ".venv\Scripts\yolo.exe"
$TFPBD  = "experiments\out\tfpbd_yolo\dataset.yaml"
$SHARED = "experiments\out\dhakaai_shared_yolo\dataset.yaml"

Add-Job "30_convert_tfpbd" "Convert TFP-BD to YOLO (stride 4, hold out Location 4)" -Marker "TFP-BD -> YOLO" {
    & $PY experiments\dhaka_vision\tfpbd_to_yolo.py --stride 4
}

Add-Job "31_remap_dhakaai" "Remap DhakaAI onto the shared 6-class vocabulary" -Marker "shared-class copy" {
    & $PY experiments\dhaka_vision\tfpbd_to_yolo.py --dhakaai-remap
}

Add-Job "32_train_tfpbd" "Train YOLOv8s on TFP-BD (val = unseen Location 4)" -Marker "Results saved to" {
    $last = "runs\detect\tfpbd_yolov8s\weights\last.pt"
    if (Test-Path $last) { & $YOLO detect train resume model=$last workers=0 }
    else {
        & $YOLO detect train model=yolov8s.pt data=$TFPBD epochs=30 imgsz=640 batch=8 `
            workers=0 device=0 amp=True cache=False patience=10 `
            project="runs\detect" name="tfpbd_yolov8s" exist_ok=True
    }
}

Add-Job "33_train_dhakaai_shared" "Train YOLOv8s on DhakaAI (shared classes)" -Marker "Results saved to" {
    $last = "runs\detect\dhakaai_shared_yolov8s\weights\last.pt"
    if (Test-Path $last) { & $YOLO detect train resume model=$last workers=0 }
    else {
        & $YOLO detect train model=yolov8s.pt data=$SHARED epochs=30 imgsz=640 batch=8 `
            workers=0 device=0 amp=True cache=False patience=10 `
            project="runs\detect" name="dhakaai_shared_yolov8s" exist_ok=True
    }
}

Add-Job "34_cross_tfpbd_to_dhakaai" "Cross-eval: TFP-BD model -> DhakaAI val" -Marker "Results saved to" {
    & $YOLO detect val model="runs\detect\tfpbd_yolov8s\weights\best.pt" data=$SHARED `
        workers=0 device=0 project="runs\val" name="tfpbd_on_dhakaai" exist_ok=True
}

Add-Job "35_cross_dhakaai_to_tfpbd" "Cross-eval: DhakaAI model -> TFP-BD val" -Marker "Results saved to" {
    & $YOLO detect val model="runs\detect\dhakaai_shared_yolov8s\weights\best.pt" data=$TFPBD `
        workers=0 device=0 project="runs\val" name="dhakaai_on_tfpbd" exist_ok=True
}

Add-Job "36_within_tfpbd" "Within-domain baseline: TFP-BD model -> TFP-BD val" -Marker "Results saved to" {
    & $YOLO detect val model="runs\detect\tfpbd_yolov8s\weights\best.pt" data=$TFPBD `
        workers=0 device=0 project="runs\val" name="tfpbd_on_tfpbd" exist_ok=True
}

Add-Job "37_within_dhakaai" "Within-domain baseline: DhakaAI model -> DhakaAI val" -Marker "Results saved to" {
    & $YOLO detect val model="runs\detect\dhakaai_shared_yolov8s\weights\best.pt" data=$SHARED `
        workers=0 device=0 project="runs\val" name="dhakaai_on_dhakaai" exist_ok=True
}

Write-Ledger $null
Write-Host ""
Write-Host ("Phase 3 run - " + $script:Jobs.Count + " jobs") -ForegroundColor Yellow
foreach ($j in $script:Jobs) { Invoke-Job $j }
Write-Ledger $null
Write-Host ""
Write-Host "ALL JOBS ATTEMPTED. See RUN_PROGRESS_PHASE3.md" -ForegroundColor Yellow
[void][Win32.Power3]::SetThreadExecutionState(2147483648)
