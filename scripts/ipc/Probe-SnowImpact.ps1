<#
.SYNOPSIS
Two-target snow impact test at a fine voxel: flat floor vs a rock (sphere collider).

.DESCRIPTION
WRITES: creates (or reuses) the matter domain "SnowTest" (1.5 x 2.5 x 1.5 m,
voxel 0.025 m, Snow) and the collider "SnowTest Rock", re-seeds a 0.3 m snow
block at y 2.0-2.3 and drops it ~2 m (about 6 m/s at contact). Run it on a
scratch scene; it does not touch other domains, but other colliders in the
box will take part.

Why two targets: J_P only reports NET volume. A flat slab on a flat floor is
laterally confined, so the impact compacts it and almost nothing loosens even
when the law is right. A rock concentrates the load and opens hoop tension:
that is where the snow law must crack. Measured 2026-10-23 (shaders of batch 1,
FLIP off): floor -> mean J_P 0.68, 13 parcels past 50 % stiffness loss;
rock -> mean J_P 1.73, 67 % past 50 % loss, 49 % fully powdered (too brittle:
no chunks survive).

What to read in the table:
  jp min     >= rho/917 (0.436 for Snow) once the batch-2 shaders are built
  hcap       ~0 (hardening clamp); thousands = stale .spv or wrong substeps
  d50 / d90  parcels past 50 % / 90 % stiffness loss: the crack and the powder
  v_max      after contact: fragments flying (FLIP) vs a dead stop (PIC)
  jp max at rest must NOT keep rising: that is the rest-jitter ratchet

.EXAMPLE
.\scripts\ipc\Probe-SnowImpact.ps1
.EXAMPLE
.\scripts\ipc\Probe-SnowImpact.ps1 -Target rock -Steps 120
#>
param(
    [ValidateSet('both', 'floor', 'rock')][string]$Target = 'both',
    [int]$Steps = 100,
    [int]$SampleEvery = 5,
    [float]$Voxel = 0.025,
    # Snow (powder, E 1.4e5) or Packed Snow (E 1e6), or any derived snow.
    [string]$Substance = 'Snow'
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
$domain = 'SnowTest'
$rock = 'SnowTest Rock'

$names = @((Invoke-RtIpc fluid.list_domains).domains | ForEach-Object { $_.name })
if ($names -notcontains $domain) {
    Invoke-RtIpc fluid.create_domain @{ name = $domain; type = 'matter';
        domain_min = @(-0.75, 0, -0.75); domain_max = @(0.75, 2.5, 0.75); voxel_size = $Voxel } | Out-Null
}
Invoke-RtIpc fluid.set_param @{ domain = $domain; default_substance = $Substance; max_particles = 300000 } | Out-Null
$d = Invoke-RtIpc fluid.get @{ domain = $domain }
Write-Host ("{0}: voxel {1} snow law {2} flip-capable build {3} E {4:N0} xi {5} ceiling {6} spread {7}" -f `
    $domain, $d.voxel_size, $d.granular_snow_plasticity, ($null -ne $d.granular_snow_max_hardening),
    $d.granular_young_modulus, $d.granular_snow_hardening, $d.granular_snow_max_hardening,
    $d.granular_snow_heterogeneity)

$colliders = @((Invoke-RtIpc collider.list).colliders | ForEach-Object { $_.name })
$fields = @('particle_count', 'granular_min_plastic_jacobian', 'granular_mean_plastic_jacobian',
    'granular_max_plastic_jacobian', 'granular_solver_substeps', 'granular_yielded',
    'granular_damage_over_50', 'granular_damage_over_90', 'granular_detached',
    'granular_hardening_capped', 'granular_invalid', 'live_mean_speed_m_s', 'live_max_speed_m_s')

function Invoke-Case([string]$name, [bool]$withRock) {
    if ($withRock) {
        if ($colliders -notcontains $rock) {
            Invoke-RtIpc collider.create @{ name = $rock; source_mode = 'sphere';
                sphere_center = @(0, 0.12, 0); sphere_radius = 0.12;
                fluid_collision_enabled = $true; friction = 0.3; restitution = 0.0 } | Out-Null
            $script:colliders += $rock
        } else {
            Invoke-RtIpc collider.update @{ name = $rock; enabled = $true } | Out-Null
        }
    } elseif ($colliders -contains $rock) {
        Invoke-RtIpc collider.update @{ name = $rock; enabled = $false } | Out-Null
    }
    Invoke-RtIpc fluid.seed @{ domain = $domain; seed_min = @(-0.15, 2.0, -0.15);
        seed_max = @(0.15, 2.3, 0.15); particles_per_cell = 8 } | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RtIpc fluid.run @{ steps = $Steps; sample_every = $SampleEvery;
        domains = @($domain); fields = $fields }
    $sw.Stop()
    Write-Host "`n== $name  (wall $([math]::Round($sw.Elapsed.TotalSeconds,1)) s, step mean $([math]::Round($r.step_ms.mean)) ms)"
    $rows = foreach ($s in $r.samples) {
        $x = $s.domains.$domain
        [pscustomobject]@{ step = $s.step; n = $x.particle_count
            jp_min = [math]::Round($x.granular_min_plastic_jacobian, 3)
            jp_mean = [math]::Round($x.granular_mean_plastic_jacobian, 3)
            jp_max = [math]::Round($x.granular_max_plastic_jacobian, 3)
            sub = $x.granular_solver_substeps; yielded = $x.granular_yielded
            d50 = $x.granular_damage_over_50; d90 = $x.granular_damage_over_90
            powder = $x.granular_detached; hcap = $x.granular_hardening_capped
            invalid = $x.granular_invalid
            v_mean = [math]::Round($x.live_mean_speed_m_s, 2)
            v_max = [math]::Round($x.live_max_speed_m_s, 2) }
    }
    $rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    $last = $rows[-1]; $mid = $rows[[int]($rows.Count * 0.6)]
    $floor = [math]::Round($d.granular_density / 917.0, 3)
    $minJp = ($rows | Measure-Object jp_min -Minimum).Minimum
    Write-Host ("  J_P floor {0}: {1}" -f $floor, $(if ($minJp -ge $floor - 0.002) { 'PASS' } else { "FAIL (min $minJp)" }))
    Write-Host ("  rest ratchet (jp_max {0} -> {1}): {2}" -f $mid.jp_max, $last.jp_max,
        $(if ($last.jp_max - $mid.jp_max -lt 0.02) { 'PASS' } else { 'FAIL still loosening at rest' }))
    Write-Host ("  breakup: {0} of {1} past 50 % loss, {2} powdered" -f $last.d50, $last.n, $last.powder)
    # ★ The counters cannot tell "cracked into clumps" from "the rock punched
    # through a block of pressureless dust": both loosen the parcels round the
    # rock. Height can. On the rock the 0.3 m block's centroid rests near
    # 0.24 + 0.15 = 0.39 m (a split block slides lower but not to 0.15); a block
    # that sank round the rock to the floor sits near 0.15-0.2 m.
    $digest = Invoke-RtIpc fluid.state_digest @{ domain = $domain }
    $cy = [math]::Round($digest.centroid[1], 3)
    # Measured 2026-10-23 without the hardening floor (E 7e5, 4.6 m/s): 0.305.
    Write-Host ("  centroid y {0} m{1}" -f $cy, $(if ($withRock) { " (resting on the rock ~0.39; each cm lower = the block sank round the rock)" } else { '' }))
}

if ($Target -in 'both', 'floor') { Invoke-Case 'FLOOR (confined: expect compaction, little loosening)' $false }
if ($Target -in 'both', 'rock')  { Invoke-Case 'ROCK (hoop tension: expect cracks AND surviving chunks)' $true }
