# Measure the scene already open in Player. Trigger bursts manually on cue, or
# pass -PopBalloons N to pop N balloons over IPC (paint.wet_demo pop) spread
# across the burst window, so the measurement does not depend on a human.
[CmdletBinding()]
param(
    [ValidateRange(1, 3600)][int]$IdleSeconds = 5,
    [ValidateRange(1, 3600)][int]$BurstSeconds = 5,
    [switch]$SkipComputeTiming,
    [switch]$Android,
    [string]$Serial,
    [ValidateRange(0, 65535)][int]$Port = 0,
    [ValidateRange(0, 64)][int]$PopBalloons = 0
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '../ipc/RtIpc.psm1') -Force
$passes = @('sim_particle_spawn', 'sim_particle_ballistic', 'sim_particle_collide',
            'paint_wet', 'paint_wet.stamp', 'paint_wet.tick', 'paint_wet.resolve',
            'paint_wet.rotate', 'paint_wet.consume', 'paint_wet.coverage', 'paint_wet.reset',
            'paint_wet.channels_compact', 'paint_wet.channels', 'paint_wet.seed_height',
            'raster_main')
$adbArgs = @()
if ($Serial) { $adbArgs = @('-s', $Serial) }
$forwardedPort = 0
$computeContexts = @()
$restoreCompute = @()
$cpuSections = @('player.advance.simulation_tick', 'paint.wet.stats_collect', 'paint.wet.submit',
                 'simulation.balloon_demo', 'simulation.timeline', 'simulation.wet_paint',
                 'viewport.particle_billboards.build', 'viewport.particle_billboards.drain',
                 'viewport.particle_billboards.upload', 'viewport.particle_billboards.vertex_write')

function Get-CpuSectionTotals {
    $totals = @{}
    foreach ($section in (Invoke-RtIpc perf.list)) {
        if ($section.name -in $cpuSections) {
            $totals[$section.name] = $section
        }
    }
    return $totals
}

function Measure-Phase([string]$Phase, [int]$Seconds, [int]$Pops = 0) {
    $popQueue = [System.Collections.Generic.Queue[int]]::new()
    if ($Pops -gt 0) {
        foreach ($b in @((Invoke-RtIpc paint.wet_demo).balloons)) {
            if ($b.inflated -and $popQueue.Count -lt $Pops) { $popQueue.Enqueue([int]$b.id) }
        }
        if ($popQueue.Count -eq 0) { throw 'No inflated balloon to pop.' }
    }
    $popped = 0
    $samples = @{}
    $supported = @{}
    foreach ($pass in $passes) {
        $samples[$pass] = [System.Collections.Generic.List[double]]::new()
        $supported[$pass] = $computeContexts -contains $false
    }
    foreach ($pass in $passes -like 'paint_wet*') {
        $supported[$pass] = $computeContexts.Count -gt 0
    }
    $lastRasterSerial = [uint64]0
    foreach ($useFallback in $computeContexts) {
        $null = Invoke-RtIpc perf.gpu_kernel_timings @{
            reset = $true
            wet_paint_fallback = $useFallback
        }
    }
    $null = Invoke-RtIpc viewport.reset_frame_timings
    $cpuStart = Get-CpuSectionTotals
    Write-Host "$Phase window ($Seconds s) starts now."
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    do {
        # Pops are spread over the first 60% of the window, one per due slot.
        if ($popQueue.Count -gt 0 -and
            $timer.ElapsedMilliseconds -ge ($popped * $Seconds * 600 / [Math]::Max($Pops, 1))) {
            $null = Invoke-RtIpc paint.wet_demo @{ operation = 'pop'; config = @{ balloon = $popQueue.Dequeue() } }
            ++$popped
        }
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
    $cpuEnd = Get-CpuSectionTotals
    $wallSeconds = $timer.Elapsed.TotalSeconds
    if ($Pops -gt 0) { Write-Host "$Phase popped $popped balloon(s) over IPC." }

    foreach ($pass in $passes) {
        $values = $samples[$pass]
        $row = [ordered]@{
            Phase = $Phase
            Pass = $pass
            Kind = 'GPU'
            Supported = $supported[$pass]
            Available = $values.Count -gt 0
            Samples = $values.Count
            MeanMs = $null
            TotalMs = $null
            PeakMs = $null
            P95Ms = $null
        }
        if ($values.Count -gt 0) {
            $stats = $values | Measure-Object -Average -Maximum -Sum
            $values.Sort()
            $rank = [int][Math]::Ceiling(0.95 * $values.Count)
            $row.MeanMs = $stats.Average
            $row.TotalMs = $stats.Sum
            $row.PeakMs = $stats.Maximum
            $row.P95Ms = $values[$rank - 1]
        }
        [pscustomobject]$row
    }

    foreach ($stage in $raster.stages) {
        if ($stage.frames_ran -le 0 -or -not $raster.frames_with_gpu) { continue }
        [pscustomobject][ordered]@{
            Phase = $Phase
            Pass = "raster.$($stage.name)"
            Kind = 'GPU stage window'
            Supported = $raster.gpu_supported
            Available = $true
            Samples = $stage.frames_ran
            MeanMs = $stage.gpu_mean_ms
            TotalMs = $null
            PeakMs = $null
            P95Ms = $stage.gpu_p95_ms
        }
    }

    foreach ($pass in $cpuSections) {
        $before = $cpuStart[$pass]
        $after = $cpuEnd[$pass]
        $count = [long]$after.count - [long]$before.count
        $total = [double]$after.total_ms - [double]$before.total_ms
        if ($count -lt 0 -or $total -lt 0) {
            throw "CPU timing counters reset during the $Phase window."
        }
        [pscustomobject][ordered]@{
            Phase = $Phase
            Pass = $pass
            Kind = 'CPU wall time'
            Supported = if ($before -or $after) { $true } else { $null }
            Available = $count -gt 0
            Samples = $count
            MeanMs = if ($count -gt 0) { $total / $count } else { $null }
            TotalMs = if ($count -gt 0) { $total } else { $null }
            # Calls per wall second; for player.advance.simulation_tick this is
            # the Player loop rate (one tick per frame).
            PerSecond = if ($wallSeconds -gt 0) { $count / $wallSeconds } else { $null }
            PeakMs = $null
            P95Ms = $null
        }
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
        if (-not $SkipComputeTiming) {
            $computeContexts += $useFallback
        }
        $enabled = -not $SkipComputeTiming
        if ([bool]$initial.enabled -ne $enabled) {
            $restoreCompute += @{ enabled = [bool]$initial.enabled; wet_paint_fallback = $useFallback }
            $null = Invoke-RtIpc perf.set_gpu_kernel_timing @{
                enabled = $enabled
                wet_paint_fallback = $useFallback
            }
        }
    }
    Write-Host 'Keep Player focused and leave the open scene idle.'
    Measure-Phase Idle $IdleSeconds
    Write-Host 'Trigger bursts in the open scene during the next window.'
    Measure-Phase Burst $BurstSeconds $PopBalloons
} finally {
    try {
        foreach ($settings in $restoreCompute) {
            $null = Invoke-RtIpc perf.set_gpu_kernel_timing $settings
        }
    } finally {
        Disconnect-RtIpc
        if ($forwardedPort -gt 0) {
            & adb @adbArgs forward --remove "tcp:$forwardedPort" | Out-Null
        }
    }
}
