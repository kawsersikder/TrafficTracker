<#
  TrafficTracker - phase 2 run: error bars + reverse-direction transfer.
  Same resumable design as run_overnight.ps1 (sentinels in .run_state2\).
  ASCII ONLY - PowerShell 5.1 reads this file as ANSI.
#>

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $Repo

$PY       = Join-Path $Repo ".venv\Scripts\python.exe"
$State    = Join-Path $Repo ".run_state2"
$Logs     = Join-Path $Repo "logs2"
$Progress = Join-Path $Repo "RUN_PROGRESS_PHASE2.md"

New-Item -ItemType Directory -Force -Path $State, $Logs, "F:\temp" | Out-Null
if (-not (Test-Path $PY)) { Write-Host "FATAL: no python at $PY" -ForegroundColor Red; pause; exit 1 }

$env:TEMP = "F:\temp"; $env:TMP = "F:\temp"
$env:PYTHONUNBUFFERED = "1"; $env:KMP_DUPLICATE_LIB_OK = "TRUE"

$sig = '[DllImport("kernel32.dll", SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags);'
try {
    Add-Type -Name Power2 -Namespace Win32 -MemberDefinition $sig -ErrorAction Stop
    [void][Win32.Power2]::SetThreadExecutionState(2147483648 -bor 1 -bor 64)
} catch { Write-Host "could not inhibit sleep: $_" -ForegroundColor Yellow }

$script:Jobs      = [System.Collections.ArrayList]::new()
$script:Results   = [ordered]@{}
$script:StartedAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

function Add-Job([string]$Name, [string]$Desc, [scriptblock]$Body, [string]$Marker) {
    [void]$script:Jobs.Add([pscustomobject]@{ Name=$Name; Desc=$Desc; Body=$Body; Marker=$Marker })
}

function Write-Ledger([string]$Current) {
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("# Phase 2 run - live progress")
    [void]$L.Add("")
    [void]$L.Add("Started: $script:StartedAt")
    [void]$L.Add("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("")
    [void]$L.Add("Purpose: seed replicates for the transfer curves (error bars on the")
    [void]$L.Add("headline result) plus reverse-direction Bangkok -> Manila transfer.")
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

$MANILA = "experiments\out\manila_segments.parquet"
$LOOP   = "experiments\out\sathorn.parquet"
$CCTV   = "experiments\out\sathorn_cctv.parquet"
$GRU1   = "experiments\runs\manila_segments_gru_cls_h1\best.pt"

Add-Job "20_seed1_transfer_cctv" "Transfer Manila -> Bangkok CCTV, seed 1" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $CCTV --pretrained $GRU1 --seed 1 --k-days 1 3 7 14 28
}

Add-Job "21_seed1_transfer_loop" "Transfer Manila -> loop-coil, seed 1" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $LOOP --pretrained $GRU1 --seed 1 --k-days 1 3 7 14 28
}

Add-Job "22_seed1_transfer_edsa" "Transfer within-Manila -> EDSA, seed 1" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $MANILA --target-prefix "EDSA|" --seed 1 --k-days 1 3 7 14 28
}
Add-Job "20_seed2_transfer_cctv" "Transfer Manila -> Bangkok CCTV, seed 2" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $CCTV --pretrained $GRU1 --seed 2 --k-days 1 3 7 14 28
}

Add-Job "21_seed2_transfer_loop" "Transfer Manila -> loop-coil, seed 2" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $LOOP --pretrained $GRU1 --seed 2 --k-days 1 3 7 14 28
}

Add-Job "22_seed2_transfer_edsa" "Transfer within-Manila -> EDSA, seed 2" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $MANILA --target-prefix "EDSA|" --seed 2 --k-days 1 3 7 14 28
}
Add-Job "23_reverse_cctv_to_manila" "Reverse transfer: Bangkok CCTV -> Manila EDSA" -Marker "runs done in" {
    & $PY experiments\transfer_kday.py --data $MANILA --target-prefix "EDSA|" --pretrained "experiments\runs\sathorn_cctv_gru_reg_h1\best.pt" --k-days 1 3 7 14 28
}

Add-Job "24_aggregate" "Regenerate tables + figures" -Marker "results_table|wrote" {
    & $PY experiments\analysis\aggregate_results.py
}

Add-Job "25_transfer_variance" "Summarise transfer error bars" -Marker "wrote|mean" {
    & $PY experiments\analysis\transfer_variance.py
}

Write-Ledger $null
Write-Host ""
Write-Host ("Phase 2 run - " + $script:Jobs.Count + " jobs") -ForegroundColor Yellow
foreach ($j in $script:Jobs) { Invoke-Job $j }
Write-Ledger $null
Write-Host ""
Write-Host "ALL JOBS ATTEMPTED. See RUN_PROGRESS_PHASE2.md" -ForegroundColor Yellow
[void][Win32.Power2]::SetThreadExecutionState(2147483648)
