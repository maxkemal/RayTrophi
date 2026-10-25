<#
.SYNOPSIS
Damp packed sand block: does it stand, and does it break apart under a load?

.DESCRIPTION
WRITES: creates (or reuses) matter domain "SandTest" (1.2 x 1.2 x 1.2 m,
voxel 0.05) and substance "WetSandProbe" (Sand, DEM, grain water capacity
0.1), seeds a 0.3 x 0.5 x 0.3 m block with fluid.seed grain_saturation and
runs it. Use a scratch scene.

★ Every case is ONE seed + ONE run. Editing a collider or flow source while
the sim runs rewinds it to frame 0 (measured 2026-10-23: a mold-removal test
came back with 0 particles), so a test that needs "remove the mold, then
watch" must be authored as an initial state. That is why fluid.seed now takes
grain_saturation: before, a damp block could only be poured.

Cases (wetness = fraction of each grain's water capacity):
  column  : free-standing block on the floor. Dry sand must slump to its
            repose angle (centroid drops a lot); damp sand must stand (centroid
            ~0.25 for a 0.5 m block).
  rock    : the same block dropped 0.6 m onto a 0.12 m sphere. Damp packed
            sand should crack into clumps, not flow round the rock (centroid
            ~0.45 resting on top; much lower = it poured round it).

.EXAMPLE
.\scripts\ipc\Probe-WetSandBlock.ps1
.EXAMPLE
.\scripts\ipc\Probe-WetSandBlock.ps1 -Wetness 0,0.1,0.3,0.8 -Case column
#>
param(
    [double[]]$Wetness = @(0.0, 0.3),
    [ValidateSet('column', 'rock', 'both')][string]$Case = 'both',
    [int]$Steps = 120
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
$domain = 'SandTest'; $sub = 'WetSandProbe'; $rock = 'SandTest Rock'

$subs = (Invoke-RtIpc substance.list @{}) | ConvertTo-Json -Depth 4 -Compress
if ($subs -notmatch "`"name`":`"$sub`"") {
    Invoke-RtIpc substance.derive @{ name = $sub; based_on = 'Sand' } | Out-Null
}
Invoke-RtIpc substance.set @{ name = $sub; fields = @{ grain_water_capacity_fraction = 0.1 } } | Out-Null
$names = @((Invoke-RtIpc fluid.list_domains).domains | ForEach-Object { $_.name })
if ($names -notcontains $domain) {
    Invoke-RtIpc fluid.create_domain @{ name = $domain; type = 'matter';
        domain_min = @(-0.6, 0, -0.6); domain_max = @(0.6, 1.2, 0.6); voxel_size = 0.05 } | Out-Null
}
Invoke-RtIpc fluid.set_param @{ domain = $domain; default_substance = $sub; max_particles = 40000 } | Out-Null
Invoke-RtIpc fluid.set_grain_settings @{ domain = $domain; radius_m = 0.0125 } | Out-Null
$probeD = Invoke-RtIpc fluid.get @{ domain = $domain }
if ($null -eq $probeD.seed_grain_saturation) { throw "This build predates fluid.seed grain_saturation. Rebuild first." }
$colliders = @((Invoke-RtIpc collider.list).colliders | ForEach-Object { $_.name })
if ($colliders -notcontains $rock) {
    Invoke-RtIpc collider.create @{ name = $rock; source_mode = 'sphere'; sphere_center = @(0, 0.12, 0);
        sphere_radius = 0.12; fluid_collision_enabled = $true; friction = 0.5; restitution = 0.0 } | Out-Null
}

function Invoke-Block([string]$label, [double]$wet, [bool]$onRock) {
    Invoke-RtIpc collider.update @{ name = $rock; enabled = $onRock } | Out-Null
    $y0 = if ($onRock) { 0.85 } else { 0.0 }
    Invoke-RtIpc fluid.seed @{ domain = $domain; seed_min = @(-0.15, $y0, -0.15);
        seed_max = @(0.15, ($y0 + 0.5), 0.15); particles_per_cell = 8; grain_saturation = $wet } | Out-Null
    $g0 = Invoke-RtIpc fluid.state_digest @{ domain = $domain }
    $r = Invoke-RtIpc fluid.run @{ steps = $Steps; sample_every = $Steps; domains = @($domain);
        fields = @('particle_count', 'live_mean_speed_m_s') }
    $g1 = Invoke-RtIpc fluid.state_digest @{ domain = $domain }
    $o = (Invoke-RtIpc fluid.matter_models @{ domain = $domain }).grain_ownership
    $x = $r.samples[-1].domains.$domain
    "{0,-8} wet {1,4}: n={2} grains={3} wet_grains={4} centroid_y {5:N3} -> {6:N3}  v_end {7:N3}  step {8} ms" -f `
        $label, $wet, $x.particle_count, $o.enabled, $o.wet_grains, $g0.centroid[1], $g1.centroid[1],
        $x.live_mean_speed_m_s, [math]::Round($r.step_ms.mean)
}

foreach ($w in $Wetness) {
    if ($Case -in 'column', 'both') { Invoke-Block 'column' $w $false }
    if ($Case -in 'rock', 'both')   { Invoke-Block 'rock' $w $true }
}
Write-Host "Read: column dry -> centroid falls well below 0.25 (slump); damp -> stays ~0.25 (stands)."
Write-Host "      rock: ~0.45+ = resting/cracked on top; near 0.1-0.2 = poured round the rock."
