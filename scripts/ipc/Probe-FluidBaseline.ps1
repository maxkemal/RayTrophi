<#
.SYNOPSIS
Collect a first fluid step baseline through the existing IPC API.

.DESCRIPTION
fluid.step advances the whole SimulationWorld and changes simulation state.
Pause playback and use an isolated scene. The script refuses multiple domains
because particle.stats.grid_domain_ms is an aggregate. It does not reset or
change authoring settings. Run each resolution from the same saved start state.

The IPC wall time includes queue and response latency. grid_domain_ms is the
internal time for all grid domains, including work outside APICFluidSolver::step.
Neither number is isolated GPU kernel time.

.EXAMPLE
.\scripts\ipc\Probe-FluidBaseline.ps1 -Domain Water -Warmup 5 -Samples 30
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Domain,
    [ValidateRange(0, 1000)][int]$Warmup = 5,
    [ValidateRange(1, 1000)][int]$Samples = 30,
    [ValidateRange(0.000001, 1.0)][double]$Dt = (1.0 / 60.0),
    [switch]$TransferStats,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

function Get-Percentile([double[]]$Values, [double]$Fraction) {
    $ordered = @($Values | Sort-Object)
    $index = [Math]::Min($ordered.Count - 1, [Math]::Ceiling($Fraction * $ordered.Count) - 1)
    return [double]$ordered[$index]
}

try {
    $domains = @((Invoke-RtIpc fluid.list_domains).domains)
    $fluidDomains = @($domains | Where-Object { $_.type -eq 'fluid' })
    $named = @($fluidDomains | Where-Object { $_.name -eq $Domain })
    if ($named.Count -ne 1) {
        throw "Expected one fluid domain named '$Domain'; found $($named.Count)."
    }
    if ($domains.Count -ne 1) {
        throw "fluid.step advances every domain. Use an isolated scene (found $($domains.Count))."
    }

    for ($i = 0; $i -lt $Warmup; $i++) {
        $null = Invoke-RtIpc fluid.step @{ dt = $Dt }
    }
    $before = Invoke-RtIpc fluid.get @{ domain = $Domain }
    if (-not $before.live_state -or $before.particle_count -le 0) {
        throw "Domain needs a live state with particles after warmup."
    }
    $memoryBefore = Invoke-RtIpc perf.get_gpu_memory
    $cacheBefore = Invoke-RtIpc sim_cache.status

    $rows = @()
    for ($i = 0; $i -lt $Samples; $i++) {
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $null = Invoke-RtIpc fluid.step @{ dt = $Dt }
        $timer.Stop()
        $particle = Invoke-RtIpc particle.stats
        $fluid = Invoke-RtIpc fluid.get @{ domain = $Domain }
        $stepStats = $null
        if ($TransferStats) {
            $stepStats = Invoke-RtIpc fluid.step_stats @{ domain = $Domain }
            if (-not $stepStats.measured) {
                throw "Fluid transfer stats were not measured at sample $i."
            }
        }
        if (-not $fluid.live_state) {
            throw "fluid.get lost its live state at sample $i."
        }
        $rows += [pscustomobject]@{
            sample = $i + 1
            ipc_wall_ms = $timer.Elapsed.TotalMilliseconds
            world_total_ms = [double]$particle.total_ms
            grid_domain_ms = [double]$particle.grid_domain_ms
            particle_count = [int64]$fluid.particle_count
            backend = [string]$fluid.backend
            step_stats = $stepStats
        }
    }

    $after = Invoke-RtIpc fluid.get @{ domain = $Domain }
    $memoryAfter = Invoke-RtIpc perf.get_gpu_memory
    $cacheAfter = Invoke-RtIpc sim_cache.status
    $gridTimes = [double[]]@($rows | ForEach-Object { $_.grid_domain_ms })
    $wallTimes = [double[]]@($rows | ForEach-Object { $_.ipc_wall_ms })
    $report = [pscustomobject]@{
        domain = $Domain
        dt = $Dt
        warmup = $Warmup
        samples = $Samples
        voxel_size = $before.voxel_size
        bounds_min = $before.domain_min
        bounds_max = $before.domain_max
        backend_before = $before.backend
        backend_after = $after.backend
        particles_before = $before.particle_count
        particles_after = $after.particle_count
        grid_domain_median_ms = Get-Percentile $gridTimes 0.5
        grid_domain_p90_ms = Get-Percentile $gridTimes 0.9
        ipc_wall_median_ms = Get-Percentile $wallTimes 0.5
        ipc_wall_p90_ms = Get-Percentile $wallTimes 0.9
        memory_before = $memoryBefore
        memory_after = $memoryAfter
        cache_before = $cacheBefore
        cache_after = $cacheAfter
        transfer_stats = [bool]$TransferStats
        rows = $rows
    }

    if ($OutputPath) {
        $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
        Write-Host "Saved $OutputPath"
    }
    $report | ConvertTo-Json -Depth 20
} finally {
    Disconnect-RtIpc
}
