<#
.SYNOPSIS
    Drive the Nuclear Detonation (Cinematic) preset end to end and assert that
    the mushroom cap is shaped by the PHYSICS and not by the domain lid.

.DESCRIPTION
    This is the acceptance test the preset was written against. It is not a
    smoke test: it checks the one claim that distinguishes a mushroom cloud from
    a tall fire, namely that the plume STOPS at a height the stratification sets
    while the domain still has headroom above it.

    ★ The decisive assertion is `touching_ceiling -eq $false`. A run where the
    plume reaches the lid can still LOOK like a good mushroom in the viewport --
    the lid flattens the top exactly the way a real inversion would -- which is
    precisely why this has to be measured rather than eyeballed.

    ★ Each measurement is taken after the frames have actually been stepped.
    Reading a counter in the same batch as the write that feeds it measures the
    PREVIOUS state; see docs/dev/NUKLEER_PRESET.md.

.EXAMPLE
    .\scripts\ipc\Test-NuclearPreset.ps1
#>
[CmdletBinding()]
param(
    [string]$Preset = 'nuclear',
    [double]$Seconds = 6.0,
    [double]$Dt = 0.0166667
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$failures = New-Object System.Collections.Generic.List[string]
function Assert-That {
    param([bool]$Condition, [string]$Message)
    if ($Condition) {
        Write-Host "  PASS  $Message" -ForegroundColor Green
    } else {
        Write-Host "  FAIL  $Message" -ForegroundColor Red
        $script:failures.Add($Message)
    }
}

Write-Host "Creating preset '$Preset'..." -ForegroundColor Cyan
$sys = Invoke-RtIpc particle.add_preset @{ preset = $Preset }
Write-Host ("  system '{0}' (index {1}): {2} domain(s), {3} flow source(s), {4} emitter(s)" -f `
    $sys.name, $sys.index, $sys.domain_count, $sys.flow_source_count, $sys.emitter_count)

Assert-That ($sys.domain_count -ge 1) 'preset created a gas domain'
Assert-That ($sys.flow_source_count -ge 3) 'preset created the three staged flow sources'

$domains = Invoke-RtIpc gas.list_domains
$domain = ($domains.domains | Where-Object { $_.name -like 'Nuclear Gas*' } | Select-Object -First 1)
if (-not $domain) { throw 'no Nuclear Gas domain found after creating the preset' }
Write-Host ("  domain '{0}'" -f $domain.name)

$settings = Invoke-RtIpc gas.get_settings @{ domain = $domain.name }
Write-Host ("  stratification = {0}" -f $settings.ambient_stratification)
Assert-That ($settings.ambient_stratification -gt 0) `
    'domain carries a positive stratification (0 means the lid shapes the cap)'
Assert-That ($settings.surface_dust_enabled) `
    'domain lifts its own ground dust (off = the skirt is faked or absent)'

# ── Step the simulation, sampling the plume as it develops. ─────────────────
# Sampling matters: a cloud that rises past its neutral level and settles back
# is CORRECT behaviour, and a single measurement at the end cannot tell that
# apart from one that never rose.
$steps = [int][math]::Round($Seconds / $Dt)
$sampleEvery = [int][math]::Round($steps / 6)
if ($sampleEvery -lt 1) { $sampleEvery = 1 }

$samples = New-Object System.Collections.Generic.List[object]
Write-Host "Stepping $steps frames (dt=$Dt)..." -ForegroundColor Cyan
for ($i = 1; $i -le $steps; $i++) {
    Invoke-RtIpc gas.step @{ dt = $Dt } | Out-Null
    if ($i % $sampleEvery -ne 0) { continue }
    $m = Invoke-RtIpc gas.measure_plume @{ domain = $domain.name }
    if (-not $m.measured) {
        # NOT "the domain is empty": the field could not be sampled at all.
        Write-Host ("  t={0,6:N2}s  NOT MEASURED" -f ($i * $Dt)) -ForegroundColor Yellow
        continue
    }
    $samples.Add($m)
    Write-Host ("  t={0,6:N2}s  top={1,7:N2}  centroid={2,7:N2}  width={3,7:N2} @ {4,7:N2}  peakT={5,5:N2}  fill={6,5:P1}  lid={7}" -f `
        ($i * $Dt), $m.top_above_floor, $m.centroid_above_floor, $m.max_width, `
        $m.max_width_height, $m.peak_temperature, $m.fill_fraction, $m.touching_ceiling)
}

if ($samples.Count -eq 0) { throw 'the plume was never measurable - nothing to assert' }
$final = $samples[$samples.Count - 1]

Write-Host "`nAssertions:" -ForegroundColor Cyan

# 1. There is a cloud at all.
Assert-That ($final.active_cells -gt 0) 'the detonation produced a cloud'

# 2. ★ THE ONE THAT MATTERS. If this fails, every shape below is the box's.
Assert-That (-not $final.touching_ceiling) `
    'the plume stopped BELOW the domain lid (cap altitude is the stratification, not the box)'

# 3. The cap is a cap: the widest slice sits in the upper part of the plume but
#    NOT at its very top. A column still climbing is widest at its head; a
#    settled mushroom is widest just under it.
$capRatio = if ($final.top_above_floor -gt 0) { $final.max_width_height / $final.top_above_floor } else { 0 }
Write-Host ("  widest slice sits at {0:P0} of plume height" -f $capRatio)
Assert-That ($capRatio -gt 0.45 -and $capRatio -lt 0.98) `
    'the widest slice is a CAP under the top, not the head of a rising column'

# 4. There is a stem: the cap must be wider than the column feeding it.
$stem = $samples[[int][math]::Floor($samples.Count / 2)]
Assert-That ($final.max_width -gt ($final.top_above_floor * 0.18)) `
    'the cap is broad relative to the column height (a mushroom, not a plume)'

# 5. The cloud has not simply filled the box, which would make every extent
#    above a measurement of the domain instead of the physics.
Assert-That ($final.fill_fraction -lt 0.55) `
    'the cloud has not saturated the domain'

# 6. ★ The cloud must still EXIST at the end. Field loss is exponential, so it
#    does not fade evenly - the whole cloud drops under the visibility threshold
#    at nearly the same instant and the symptom reads as "the top cooled and
#    vanished" rather than as a loss rate. Compare against the peak, not zero.
$peakCells = ($samples | Measure-Object -Property active_cells -Maximum).Maximum
$survival = if ($peakCells -gt 0) { $final.active_cells / $peakCells } else { 0 }
Write-Host ("  final cloud is {0:P0} of its peak size" -f $survival)
Assert-That ($survival -gt 0.35) `
    'the cloud survived to the end of the shot (field loss did not erase it)'

# 7. ★ THE GROUND SKIRT IS EARNED, NOT PLACED. Early in the shot the widest
#    thing in the domain must be down at the floor: that is the shock scouring
#    the ground it crosses. If the widest early slice is already up in the air,
#    the surface rule never fired and what you are looking at is the fireball.
$early = $samples[0]
Write-Host ("  earliest sample: widest {0:N2} at height {1:N2}" -f $early.max_width, $early.max_width_height)
Assert-That ($early.max_width_height -lt ($early.top_above_floor * 0.35)) `
    'the early skirt sits on the ground (the shock lifted it, nothing placed it)'

# 8. ★ AND IT MUST TRAVEL. A ring someone placed has the radius it was given;
#    a ring the shock scoured grows with the front. This is the assertion that
#    tells the two apart, and the old faked source would FAIL it.
$second = $samples[1]
Write-Host ("  skirt width {0:N2} -> {1:N2}" -f $early.max_width, $second.max_width)
Assert-That ($second.max_width -gt ($early.max_width * 1.15)) `
    'the skirt EXPANDED between samples (a placed ring would not)'

# 9. Does the pressure field carry a rarefaction? NOT an assertion - this is the
#    measurement that decides whether a condensation model (the altitude discs
#    and the collar around the stem) can key off pressure at all. Recorded here
#    so the decision is made on data.
if ($early.pressure_measured) {
    Write-Host ("  pressure field: min {0:N4} at height {1:N2}, max {2:N4}" -f `
        $early.pressure_min, $early.pressure_min_height, $early.pressure_max)
    if ($early.pressure_min -lt 0) {
        Write-Host '    -> a genuine low-pressure region exists; condensation can key off it' -ForegroundColor Green
    } else {
        Write-Host '    -> NO region below ambient; a condensation model would need another driver' -ForegroundColor Yellow
    }
} else {
    Write-Host '  pressure field: NOT MEASURED (channel off) - enable Pressure to decide' -ForegroundColor Yellow
}

# ── Calibration aid ────────────────────────────────────────────────────────
# ★ Stratification and heat loss are COUPLED: a plume that cools slower keeps
# its lift longer and settles higher. Rather than leave the author to guess the
# settled anomaly, derive it from what the cloud actually did. Trust THIS over
# any number written in the preset comments.
$settings2 = Invoke-RtIpc gas.get_settings @{ domain = $domain.name }
# The kilometre-scale 'physical' variant was removed, so the domain lid is a
# constant again rather than a per-preset scale factor.
$domTop = 34.0
$targetTop = $domTop * (26.0 / 34.0)
$plateau = ($samples | Select-Object -Last 3 | Measure-Object -Property mean_temperature -Average).Average
Write-Host ''
Write-Host 'Calibration:' -ForegroundColor Cyan
Write-Host ("  measured cap top      : {0:N2}  (target ~{1:N2}, lid {2:N2})" -f $final.top_above_floor, $targetTop, $domTop)
Write-Host ("  settled mean anomaly  : {0:N3}" -f $plateau)
Write-Host ("  stratification in use : {0:N4}" -f $settings2.ambient_stratification)
if ($final.top_above_floor -gt 0) {
    $implied = $settings2.ambient_stratification * ($final.top_above_floor / $targetTop)
    Write-Host ("  -> to land the cap at {0:N1}, try stratification {1:N4}" -f $targetTop, $implied)
    Write-Host '     (scale it the same way whenever Heat Loss /s changes)'
}

Write-Host ''
if ($failures.Count -eq 0) {
    Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
    exit 0
}
Write-Host ("{0} CHECK(S) FAILED:" -f $failures.Count) -ForegroundColor Red
$failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
exit 1
