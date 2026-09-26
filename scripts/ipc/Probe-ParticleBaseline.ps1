<#
.SYNOPSIS
Particle roadmap Phase 0 baseline: CPU vs current partial-GPU cost and
trajectory agreement, at several particle counts, over IPC.

.DESCRIPTION
For every (scenario, count, policy) the script builds the same particle cloud
from fixed-seed burst emitters, steps it, and records per-step timings, which
backend each stage ran on, and the particle step's own GPU transfers
(particle.stats). After stepping it samples particle state
(particle.get_state_sample) so the CPU and Auto runs of the same scenario can be
compared particle by particle.

Scenarios:
  ballistic       gravity only, no collider
  plane           a shared plane collider at y = 0 (collider.create)
  self_collision  ballistic + CPU self-collision

REQUIREMENTS
  * Nothing about the open scene: the script creates its OWN particle system
    and addresses every particle call to it by system_id (particle roadmap
    Phase 1), then removes it and restores the previously active system. The
    shared collider for the 'plane' scenario goes to the ACTIVE system, so the
    baseline system is active while the script runs.
  * Timeline STOPPED. particle.step advances the same runtime the frame loop
    steps; if both step, every number here is wrong. The script checks this
    (two stats reads with no step between them must match) and refuses.

READING THE RESULT
  * gpu_force_status must be 'gpu' on the Auto rows for them to mean
    anything. Any other value names why the CPU ran the force stage, and the
    Auto row is then just a second CPU row.
  * force_download_bytes per step is the cost the roadmap wants gone
    (velocity readback). mirror_upload_bytes is the end-of-step host->device
    copy of the whole SoA. Both are CAPACITY-sized, not alive-sized.
  * compare.max_position_delta is the largest CPU-vs-Auto position difference
    over the sampled particles. Small (float noise, ~1e-4 m after a second) is
    expected; large means the GPU force kernel and the CPU path disagree.

.EXAMPLE
.\scripts\ipc\Probe-ParticleBaseline.ps1 -OutputPath .\particle_baseline.json
.\scripts\ipc\Probe-ParticleBaseline.ps1 -Counts 2048 -Scenarios ballistic -Samples 10
#>
[CmdletBinding()]
param(
    [int[]]$Counts = @(1024, 8192, 32768),
    [ValidateSet('ballistic', 'plane', 'self_collision')]
    [string[]]$Scenarios = @('ballistic', 'plane', 'self_collision'),
    [ValidateSet('cpu', 'auto', 'gpu_required')]
    [string[]]$Policies = @('cpu', 'auto'),
    [ValidateRange(1, 1000)][int]$Warmup = 3,
    [ValidateRange(1, 1000)][int]$Samples = 30,
    [ValidateRange(0.000001, 1.0)][double]$Dt = (1.0 / 60.0),
    [int]$Seed = 1234,
    [ValidateRange(1, 4096)][int]$SampleParticles = 256,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$BurstPerEmitter = 512          # the runtime clamps one emitter's spawn to 512 per step
$ColliderName = 'ParticleBaselinePlane'

function Get-Percentile([double[]]$Values, [double]$Fraction) {
    if ($Values.Count -eq 0) { return 0.0 }
    $ordered = @($Values | Sort-Object)
    $index = [Math]::Min($ordered.Count - 1, [Math]::Max(0, [Math]::Ceiling($Fraction * $ordered.Count) - 1))
    return [double]$ordered[$index]
}

function Get-Median([object[]]$Rows, [string]$Field) {
    return Get-Percentile ([double[]]@($Rows | ForEach-Object { [double]$_.$Field })) 0.5
}

# Every particle call targets the baseline system, never the scene's own.
$script:BaselineSystemId = -1
function Invoke-P([string]$Method, [hashtable]$Params = @{}) {
    $p = @{} + $Params
    $p['system_id'] = $script:BaselineSystemId
    return Invoke-RtIpc $Method $p
}

function Assert-QuietRuntime {
    # Two reads with no step between them must be identical. If the frame loop
    # (timeline playback) is also stepping this runtime, total_ms/alive change.
    $a = Invoke-P particle.stats
    Start-Sleep -Milliseconds 300
    $b = Invoke-P particle.stats
    if ($a.total_ms -ne $b.total_ms -or $a.alive_count -ne $b.alive_count) {
        throw "Something else is stepping the particle runtime (timeline playing?). Stop playback and rerun."
    }
}

# ballistic/self_collision throw the cloud UP from y=3. The plane scenario
# throws it DOWN from just above the y=0 plane: thrown up, it never reached the
# plane within the measured steps and the 'plane' rows were a second ballistic
# run (identical centroid, zero contacts) -- measured 2026-09-26.
function New-BaselineCloud([int]$Count, [int]$SeedBase, [double]$Height = 3.0, [double]$DirY = 1.0) {
    $emitters = [int][Math]::Ceiling($Count / [double]$BurstPerEmitter)
    $side = [int][Math]::Ceiling([Math]::Sqrt($emitters))
    $remaining = $Count
    for ($i = 0; $i -lt $emitters; $i++) {
        $burst = [Math]::Min($BurstPerEmitter, $remaining)
        $remaining -= $burst
        $x = ($i % $side) * 1.0 - ($side * 0.5)
        $z = [Math]::Floor($i / $side) * 1.0 - ($side * 0.5)
        $null = Invoke-P particle.add_emitter @{
            name             = "Baseline_$i"
            point            = @($x, $Height, $z)
            direction        = @(0.0, $DirY, 0.0)
            rate_per_second  = 0.0
            burst_count      = $burst
            speed            = 4.0
            spread           = 0.5
            lifetime_seconds = 1000.0
            seed             = ($SeedBase + $i)
        }
    }
}

function Clear-Baseline {
    $null = Invoke-P particle.clear_emitters
    $null = Invoke-P particle.clear
}

$previousActiveId = $null
$createdCollider = $false
$physicsBefore = $null
$runs = @()

try {
    $systems = @((Invoke-RtIpc particle.list_systems).systems)
    $previous = $systems | Where-Object { $_.active } | Select-Object -First 1
    if ($previous) { $previousActiveId = [int]$previous.id }
    $created = Invoke-RtIpc particle.add_system @{ name = "Particle Baseline $(Get-Date -Format HHmmss)" }
    $script:BaselineSystemId = [int]$created.id
    Assert-QuietRuntime
    $physicsBefore = Invoke-P particle.get_physics

    foreach ($scenario in $Scenarios) {
        foreach ($count in $Counts) {
            foreach ($policy in $Policies) {
                Clear-Baseline
                $null = Invoke-P particle.set_physics @{
                    mode                   = 'spark'
                    execution_policy       = $policy
                    self_collision_enabled = ($scenario -eq 'self_collision')
                    particle_radius        = 0.04
                    gravity_scale          = 1.0
                }
                if ($scenario -eq 'plane' -and -not $createdCollider) {
                    $null = Invoke-RtIpc collider.create @{
                        name        = $ColliderName
                        source_mode = 'plane'
                        plane_y     = 0.0
                        restitution = 0.3
                        friction    = 0.2
                    }
                    $createdCollider = $true
                } elseif ($scenario -ne 'plane' -and $createdCollider) {
                    $null = Invoke-RtIpc collider.remove @{ name = $ColliderName }
                    $createdCollider = $false
                }

                if ($scenario -eq 'plane') { New-BaselineCloud $count $Seed 0.3 -1.0 }
                else { New-BaselineCloud $count $Seed }
                for ($i = 0; $i -lt $Warmup; $i++) {
                    $null = Invoke-P particle.step @{ dt = $Dt }
                }
                $afterWarmup = Invoke-P particle.stats
                if ($afterWarmup.alive_count -lt $count) {
                    Write-Warning "$scenario/$count/${policy}: only $($afterWarmup.alive_count) of $count particles alive after warmup."
                }

                $rows = @()
                for ($i = 0; $i -lt $Samples; $i++) {
                    $timer = [System.Diagnostics.Stopwatch]::StartNew()
                    $null = Invoke-P particle.step @{ dt = $Dt }
                    $timer.Stop()
                    $s = Invoke-P particle.stats
                    $rows += [pscustomobject]@{
                        ipc_wall_ms          = $timer.Elapsed.TotalMilliseconds
                        total_ms             = [double]$s.total_ms
                        emit_ms              = [double]$s.emit_ms
                        integrate_ms         = [double]$s.integrate_ms
                        self_collision_ms    = [double]$s.self_collision_ms
                        upload_ms            = [double]$s.upload_ms
                        gpu_force_ms         = [double]$s.gpu_force_ms
                        force_upload_bytes   = [double]$s.force_upload_bytes
                        force_download_bytes = [double]$s.force_download_bytes
                        force_sync_calls     = [double]$s.force_synchronize_calls
                        force_sync_ms        = [double]$s.force_synchronize_ms
                        force_download_ms    = [double]$s.force_download_call_ms
                        mirror_upload_bytes  = [double]$s.mirror_upload_bytes
                        mirror_upload_ms     = [double]$s.mirror_upload_call_ms
                        gpu_force_status     = [string]$s.gpu_force_status
                        step_blocked         = [bool]$s.step_blocked
                        nonfinite            = [int]$s.nonfinite_particles
                        alive                = [int]$s.alive_count
                    }
                }
                $last = Invoke-P particle.stats
                $stride = [Math]::Max(1, [int][Math]::Floor($last.alive_count / [double]$SampleParticles))
                $state = Invoke-P particle.get_state_sample @{ max_count = $SampleParticles; stride = $stride }
                if ($scenario -eq 'plane') {
                    # The cloud was thrown down at the plane: it must have been
                    # stopped by it. bounds_min.y far below 0 = tunneling; a
                    # centroid still above the start height = it never got there.
                    $minY = [double]$state.bounds_min[1]
                    if ($minY -lt -0.05) {
                        Write-Warning "plane/$count/${policy}: particles below the plane (min y = $minY) -- tunneling."
                    }
                    if ([double]$state.centroid[1] -gt 0.3) {
                        Write-Warning "plane/$count/${policy}: cloud centroid y = $($state.centroid[1]) is above its start height; it never reached the plane, the row measures no collision."
                    }
                }

                $statuses = @($rows | ForEach-Object { $_.gpu_force_status } | Sort-Object -Unique)
                $runs += [pscustomobject]@{
                    scenario             = $scenario
                    count                = $count
                    policy               = $policy
                    compute_backend      = $last.compute_backend
                    gpu_force_status     = ($statuses -join ',')
                    stage_backends       = $last.stage_backends
                    blocked_steps        = @($rows | Where-Object { $_.step_blocked }).Count
                    alive                = $last.alive_count
                    capacity             = $last.capacity
                    total_median_ms      = Get-Median $rows 'total_ms'
                    total_p90_ms         = Get-Percentile ([double[]]@($rows | ForEach-Object { $_.total_ms })) 0.9
                    emit_median_ms       = Get-Median $rows 'emit_ms'
                    integrate_median_ms  = Get-Median $rows 'integrate_ms'
                    self_collision_median_ms = Get-Median $rows 'self_collision_ms'
                    gpu_force_median_ms  = Get-Median $rows 'gpu_force_ms'
                    force_sync_median_ms = Get-Median $rows 'force_sync_ms'
                    force_download_median_ms = Get-Median $rows 'force_download_ms'
                    upload_median_ms     = Get-Median $rows 'upload_ms'
                    force_upload_bytes_per_step   = Get-Median $rows 'force_upload_bytes'
                    force_download_bytes_per_step = Get-Median $rows 'force_download_bytes'
                    force_sync_calls_per_step     = Get-Median $rows 'force_sync_calls'
                    mirror_upload_bytes_per_step  = Get-Median $rows 'mirror_upload_bytes'
                    ipc_wall_median_ms   = Get-Median $rows 'ipc_wall_ms'
                    nonfinite_max        = (@($rows | ForEach-Object { $_.nonfinite }) | Measure-Object -Maximum).Maximum
                    state                = $state
                    rows                 = $rows
                }
                Write-Host ("{0,-15} {1,7} {2,-12} total {3,8:N3} ms  force[{4}] {5,8:N3} ms  down {6,10:N0} B  mirror {7,10:N0} B" -f `
                    $scenario, $count, $policy, $runs[-1].total_median_ms, $runs[-1].gpu_force_status,
                    $runs[-1].gpu_force_median_ms, $runs[-1].force_download_bytes_per_step,
                    $runs[-1].mirror_upload_bytes_per_step)
            }
        }
    }

    # CPU vs every other policy, same scenario and count. Same seed + serial
    # reset (particle.clear) + same slot order means the same particle sits at
    # the same sampled index in both runs.
    $comparisons = @()
    foreach ($group in ($runs | Group-Object scenario, count)) {
        $cpu = @($group.Group | Where-Object { $_.policy -eq 'cpu' })
        if ($cpu.Count -ne 1) { continue }
        foreach ($other in @($group.Group | Where-Object { $_.policy -ne 'cpu' })) {
            $a = $cpu[0].state
            $b = $other.state
            $maxDelta = 0.0
            $matched = 0
            $n = [Math]::Min([int]$a.returned, [int]$b.returned)
            for ($i = 0; $i -lt $n; $i++) {
                if ($a.indices[$i] -ne $b.indices[$i]) { continue }
                $pa = $a.positions[$i]; $pb = $b.positions[$i]
                $d = [Math]::Sqrt([Math]::Pow($pa[0] - $pb[0], 2) + [Math]::Pow($pa[1] - $pb[1], 2) + [Math]::Pow($pa[2] - $pb[2], 2))
                if ($d -gt $maxDelta) { $maxDelta = $d }
                $matched++
            }
            $ca = $a.centroid; $cb = $b.centroid
            $centroidDelta = [Math]::Sqrt([Math]::Pow($ca[0] - $cb[0], 2) + [Math]::Pow($ca[1] - $cb[1], 2) + [Math]::Pow($ca[2] - $cb[2], 2))
            # A ratio of two CPU runs is noise, not a speedup: leave it empty
            # unless the other policy's force stage actually ran on the GPU.
            $speedup = $null
            if ($other.gpu_force_status -eq 'gpu' -and $other.total_median_ms -gt 0) {
                $speedup = $cpu[0].total_median_ms / $other.total_median_ms
            }
            $comparisons += [pscustomobject]@{
                scenario             = $cpu[0].scenario
                count                = $cpu[0].count
                policy               = $other.policy
                other_force_status   = $other.gpu_force_status
                matched_particles    = $matched
                max_position_delta   = $maxDelta
                centroid_delta       = $centroidDelta
                cpu_total_median_ms  = $cpu[0].total_median_ms
                other_total_median_ms = $other.total_median_ms
                speedup              = $speedup
            }
        }
    }

    $report = [pscustomobject]@{
        probe       = 'Probe-ParticleBaseline'
        dt          = $Dt
        warmup      = $Warmup
        samples     = $Samples
        seed        = $Seed
        runs        = $runs
        comparisons = $comparisons
    }
    if ($OutputPath) {
        $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
        Write-Host "Saved $OutputPath"
    }
    Write-Host ''
    Write-Host 'CPU vs other policy:'
    $comparisons | Format-Table scenario, count, policy, other_force_status, matched_particles,
        max_position_delta, centroid_delta, cpu_total_median_ms, other_total_median_ms, speedup -AutoSize
    $notGpu = @($comparisons | Where-Object { $_.other_force_status -ne 'gpu' })
    if ($notGpu.Count -gt 0) {
        $reasons = @($notGpu | ForEach-Object { $_.other_force_status } | Sort-Object -Unique) -join ', '
        Write-Warning ("$($notGpu.Count) of $($comparisons.Count) comparison rows never ran the GPU force stage " +
            "($reasons). Their timings compare CPU with CPU; speedup is left empty.")
    }
} finally {
    try {
        Clear-Baseline
        if ($createdCollider) { $null = Invoke-RtIpc collider.remove @{ name = $ColliderName } }
        if ($physicsBefore) {
            $null = Invoke-P particle.set_physics @{
                mode                   = $physicsBefore.mode
                execution_policy       = $physicsBefore.execution_policy
                self_collision_enabled = $physicsBefore.self_collision_enabled
                particle_radius        = $physicsBefore.particle_radius
                gravity_scale          = $physicsBefore.gravity_scale
            }
        }
        if ($script:BaselineSystemId -ge 0) {
            $null = Invoke-RtIpc particle.remove_system @{ system_id = $script:BaselineSystemId }
        }
        if ($null -ne $previousActiveId) {
            $null = Invoke-RtIpc particle.set_active_system @{ system_id = $previousActiveId }
        }
    } catch {
        Write-Warning "Cleanup failed: $_"
    }
    Disconnect-RtIpc
}
