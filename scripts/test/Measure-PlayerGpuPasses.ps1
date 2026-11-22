# Measure the scene already open in Player. Trigger bursts manually on cue.
[CmdletBinding()]
param(
    [ValidateRange(1, 3600)][int]$IdleSeconds = 5,
    [ValidateRange(1, 3600)][int]$BurstSeconds = 5,
    [switch]$Android,
    [string]$Serial,
    [ValidateRange(0, 65535)][int]$Port = 0
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '../ipc/RtIpc.psm1') -Force
$passes = @('sim_particle_spawn', 'sim_particle_ballistic', 'sim_particle_collide',
            'paint_wet', 'raster_main')
$adbArgs = @()
if ($Serial) { $adbArgs = @('-s', $Serial) }
$forwardedPort = 0
$computeContexts = @()
$restoreCompute = @()

function Measure-Phase([string]$Phase, [int]$Seconds) {
    $samples = @{}
    $supported = @{}
    foreach ($pass in $passes) {
        $samples[$pass] = [System.Collections.Generic.List[double]]::new()
        $supported[$pass] = $computeContexts -contains $false
    }
    $supported['paint_wet'] = $computeContexts.Count -gt 0
    $lastRasterSerial = [uint64]0
    foreach ($useFallback in $computeContexts) {
        $null = Invoke-RtIpc perf.gpu_kernel_timings @{
            reset = $true
            wet_paint_fallback = $useFallback
        }
    }
    $null = Invoke-RtIpc viewport.reset_frame_timings
    Write-Host "$Phase window ($Seconds s) starts now."
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    do {
        $remainingMs = ($Seconds * 1000) - $timer.ElapsedMilliseconds
        if ($remainingMs -gt 0) {
            Start-Sleep -Milliseconds ([Math]::Min(250, $remainingMs))
        }

        foreach ($useFallback in $computeContexts) {
            $compute = Invoke-RtIpc perf.gpu_kernel_timings @{
                reset = $true
                wet_paint_fallback = $useFallback
            }
            if (-not $compute.supported -or -not $compute.enabled) {
                throw 'Compute timing changed during capture. Keep the same scene open.'
            }
            foreach ($kernel in $compute.kernels) {
                if (-not $samples.ContainsKey($kernel.kernel)) { continue }
                if (@($kernel.samples_ms).Count -ne $kernel.calls) {
                    throw "Missing dispatch samples for $($kernel.kernel)."
                }
                foreach ($ms in $kernel.samples_ms) {
                    $samples[$kernel.kernel].Add([double]$ms)
                }
            }
        }

        $raster = Invoke-RtIpc viewport.frame_timings
        $supported['raster_main'] = $raster.gpu_supported
        $newSamples = $raster.raster_main_gpu_samples |
            Where-Object { [uint64]$_.frame_serial -gt $lastRasterSerial } |
            Sort-Object { [uint64]$_.frame_serial }
        foreach ($sample in $newSamples) {
            $samples['raster_main'].Add([double]$sample.ms)
            $lastRasterSerial = [uint64]$sample.frame_serial
        }
    } while ($timer.Elapsed.TotalSeconds -lt $Seconds)

    foreach ($pass in $passes) {
        $values = $samples[$pass]
        $row = [ordered]@{
            Phase = $Phase
            Pass = $pass
            Supported = $supported[$pass]
            Available = $values.Count -gt 0
            Samples = $values.Count
            MeanMs = $null
            PeakMs = $null
            P95Ms = $null
        }
        if ($values.Count -gt 0) {
            $stats = $values | Measure-Object -Average -Maximum
            $values.Sort()
            $rank = [int][Math]::Ceiling(0.95 * $values.Count)
            $row.MeanMs = $stats.Average
            $row.PeakMs = $stats.Maximum
            $row.P95Ms = $values[$rank - 1]
        }
        [pscustomobject]$row
    }
}

try {
    if ($Android -and $Port -eq 0) {
        $result = & adb @adbArgs forward tcp:0 localabstract:RayTrophiPlayer
        if ($LASTEXITCODE -ne 0) { throw 'adb forward failed.' }
        $forwardedPort = [int]$result
        if ($forwardedPort -le 0) { throw 'adb forward returned no TCP port.' }
        $Port = $forwardedPort
    }
    if ($Port -gt 0) {
        Connect-RtIpc -Port $Port
    } else {
        Connect-RtIpc -PipeName RayTrophiPlayer
    }

    foreach ($useFallback in @($false, $true)) {
        $initial = Invoke-RtIpc perf.gpu_kernel_timings @{ wet_paint_fallback = $useFallback }
        if (-not $initial.supported) { continue }
        $computeContexts += $useFallback
        if (-not $initial.enabled) {
            $null = Invoke-RtIpc perf.set_gpu_kernel_timing @{
                enabled = $true
                wet_paint_fallback = $useFallback
            }
            $restoreCompute += $useFallback
        }
    }
    Write-Host 'Keep Player focused and leave the open scene idle.'
    Measure-Phase Idle $IdleSeconds
    Write-Host 'Trigger bursts in the open scene during the next window.'
    Measure-Phase Burst $BurstSeconds
} finally {
    try {
        foreach ($useFallback in $restoreCompute) {
            $null = Invoke-RtIpc perf.set_gpu_kernel_timing @{
                enabled = $false
                wet_paint_fallback = $useFallback
            }
        }
    } finally {
        Disconnect-RtIpc
        if ($forwardedPort -gt 0) {
            & adb @adbArgs forward --remove "tcp:$forwardedPort" | Out-Null
        }
    }
}
