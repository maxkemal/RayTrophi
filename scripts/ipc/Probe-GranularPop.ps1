<#
.SYNOPSIS
Read-only probe: what IS the small burst a soft granular pile releases?

.DESCRIPTION
Play the simulation in the app; this polls fluid.get and prints one row per
sample. It NEVER writes: no set_param, no seed, no step, no reset. Safe to point
at a scene you are working in. Ctrl+C to stop and get the verdict.

A soft pile (E ~ 1 kPa, no cohesion) settles like wet mud and then, every so
often, lets something go with a small pop. Three mechanisms produce that same
picture and the eye cannot separate them:

  (a) A REAL compaction pocket. K = E/(3(1-2nu)) is a few hundred Pa, so the
      material must compress enormously before it pushes back. Regions load up
      and relieve. At E = 1 kPa this is the honest answer.
  (b) The dt*C CLAMP in the stress kernel (flag bit 16) -- the subcycle failed
      to cover the motion and the shader held the step together. Numerical.
  (c) The det(F) RESET (flag bit 4) -- a particle compacted past the kernel's
      floor, so its stress was dumped to zero in one step. Numerical, and a
      step-function discharge, which is what actually reads as a "pop".

The discriminator is COINCIDENCE IN TIME, not magnitude. A burst on the same
sample as a jump in INVALID or CLAMPED names itself. A burst with both counters
flat means the physics is doing it, and only a stiffer E changes it.

A counter reading zero every sample is also a result: it is what rules (b) and
(c) out. Do not stop polling because "nothing is happening".

.EXAMPLE
.\scripts\ipc\Probe-GranularPop.ps1
.EXAMPLE
.\scripts\ipc\Probe-GranularPop.ps1 -Domain SandBox -IntervalMs 200
#>
param(
    [string]$Domain,
    [int]$IntervalMs = 250
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$domains = (Invoke-RtIpc fluid.list_domains).domains
if (-not $Domain) {
    $granular = @($domains | Where-Object { $_.granular_enabled })
    if ($granular.Count -eq 0) {
        throw "No granular domain in the scene. Present: $(($domains.name) -join ', ')"
    }
    if ($granular.Count -gt 1) {
        throw "Several granular domains; pass -Domain. Found: $(($granular.name) -join ', ')"
    }
    $Domain = $granular[0].name
}

$state = Invoke-RtIpc fluid.get @{ domain = $Domain }
if ($null -eq $state.granular_strain_limited) {
    throw "This build predates the strain-rate instrumentation (granular_strain_limited missing). Rebuild first."
}

Write-Host ("domain: {0}" -f $Domain)
Write-Host ("E requested={0:N0} Pa  effective={1:N0} Pa  cohesion={2:N0} Pa" -f `
    $state.granular_requested_young_modulus, $state.granular_effective_young_modulus, $state.granular_cohesion)
Write-Host ("overburden={0:N0} Pa  E needed for small strain={1:N0} Pa  below_load={2}" -f `
    $state.granular_overburden_pressure, $state.granular_young_modulus_for_load, $state.granular_stiffness_below_load)
Write-Host ""
Write-Host "  time   parts   |C|   sub(w/s/run)  yield  detach  INVALID  CLAMPED   note"
Write-Host ("  " + ("-" * 74))

$started = Get-Date
$prevRate = $null; $prevInvalid = 0; $prevClamped = 0
$samples = 0; $invalidEvents = 0; $clampEvents = 0; $quietBursts = 0; $peakRate = 0.0

try {
    while ($true) {
        $s = Invoke-RtIpc fluid.get @{ domain = $Domain }
        $rate = [double]$s.granular_strain_rate
        if ($rate -gt $peakRate) { $peakRate = $rate }
        $invalid = [int]$s.granular_invalid
        $clamped = [int]$s.granular_strain_limited

        $note = ""
        if ($null -ne $prevRate) {
            $threshold = [Math]::Max(2.0 * $prevRate, 1.0)
            $burst = $rate -gt $threshold
            $grewInvalid = $invalid -gt $prevInvalid
            $grewClamped = $clamped -gt $prevClamped
            if ($grewInvalid) { $invalidEvents++ }
            if ($grewClamped) { $clampEvents++ }
            if ($burst -and $grewInvalid)      { $note = "<< POP = det(F) RESET (c) - numerical" }
            elseif ($burst -and $grewClamped)  { $note = "<< POP = dt*C CLAMP (b) - subcycle too coarse" }
            elseif ($burst)                    { $quietBursts++; $note = "<< POP with both counters flat = COMPACTION (a)" }
            elseif ($grewInvalid)              { $note = "det(F) reset (no visible burst)" }
            elseif ($grewClamped)              { $note = "dt*C clamped (no visible burst)" }
        }

        Write-Host ("  {0,5:N1}s  {1,6}  {2,5:N1}   {3,2}/{4,2}/{5,2}      {6,5}  {7,6}  {8,7}  {9,7}   {10}" -f `
            ((Get-Date) - $started).TotalSeconds, $s.particle_count, $rate,
            $s.granular_wave_substeps, $s.granular_strain_substeps, $s.granular_solver_substeps,
            $s.granular_yielded, $s.granular_detached, $invalid, $clamped, $note)

        $prevRate = $rate; $prevInvalid = $invalid; $prevClamped = $clamped
        $samples++
        Start-Sleep -Milliseconds $IntervalMs
    }
}
finally {
    Write-Host ""
    Write-Host ("samples={0}  peak |C|={1:N1} 1/s" -f $samples, $peakRate)
    Write-Host ("det(F) resets on {0} samples, dt*C clamps on {1}, bursts with neither on {2}" -f `
        $invalidEvents, $clampEvents, $quietBursts)
    if ($invalidEvents -eq 0 -and $clampEvents -eq 0) {
        Write-Host "VERDICT: no numerical discharge in this window. The pops are compaction relief (a) - the material really is this soft, and only a stiffer E changes it."
    } elseif ($invalidEvents -gt 0) {
        Write-Host "VERDICT: det(F) resets are firing. Stress is dumped to zero in one step. Raise E toward granular_young_modulus_for_load; if that is not acceptable, the compaction floor in sim_fluid_granular_stress_update.comp needs a real volumetric plasticity cap instead of a reset."
    } else {
        Write-Host "VERDICT: the dt*C clamp is carrying the step. Raise granular_max_solver_substeps until CLAMPED stays 0."
    }
    Disconnect-RtIpc
}
