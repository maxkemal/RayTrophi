<#
.SYNOPSIS
Drives a Snow drop and reads whether the Stomakhin 2013 law is actually running.

.DESCRIPTION
WRITES: re-seeds the domain's box (fluid.seed, replace=true) and steps it with
fluid.run. Use it on a test scene, not one you are authoring in.

What it separates, in order:
  1. Counters measured at all (granular_material_measured). Before 2026-10-23
     the Matter path never counted, so a 4 m/s impact read 0 yielded for 120
     steps. False here means the counting fix is not in this binary.
  2. Law selected (granular_snow_plasticity). False = the Default Substance is
     still Drucker-Prager; nothing below means anything.
  3. J_P leaves 1 in BOTH directions. Compaction (min < 1) is what makes a
     clump; loosening (max > 1) is what makes a crack. A run whose J_P only
     goes down compacts and never breaks -- the chocolate symptom again.
  4. Hardening clamp stays rare. A large, persistent count means the subcycle
     is too coarse for how compacted the pack is (stiffness silently capped).

.EXAMPLE
.\scripts\ipc\Probe-SnowStomakhin.ps1
.EXAMPLE
.\scripts\ipc\Probe-SnowStomakhin.ps1 -Domain "Physics Domain 1" -Steps 180 -SampleEvery 15
#>
param(
    [string]$Domain,
    [int]$Steps = 120,
    [int]$SampleEvery = 10,
    [int]$ParticlesPerCell = 8
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$domains = (Invoke-RtIpc fluid.list_domains).domains
if (-not $Domain) {
    $granular = @($domains | Where-Object { $_.granular_enabled })
    if ($granular.Count -ne 1) {
        throw "Pass -Domain. Granular domains: $(($granular.name) -join ', ')"
    }
    $Domain = $granular[0].name
}
$d = Invoke-RtIpc fluid.get @{ domain = $Domain }
if ($null -eq $d.granular_material_measured) {
    throw "This build predates the snow counters (granular_material_measured missing). Rebuild first."
}
Write-Host ("domain {0}  substance {1}  snow law {2}  E {3:N0} Pa  rho {4:N0}  theta_c {5}  theta_s {6}  xi {7}" -f `
    $Domain, $d.default_substance, $d.granular_snow_plasticity, $d.granular_young_modulus,
    $d.granular_density, $d.granular_snow_critical_compression,
    $d.granular_snow_critical_stretch, $d.granular_snow_hardening)
if (-not $d.granular_snow_plasticity) {
    Write-Warning "Default Substance is not on the snow law; J_P below is Drucker-Prager compaction."
}

Invoke-RtIpc fluid.seed @{ domain = $Domain; seed_min = $d.seed_min; seed_max = $d.seed_max;
    particles_per_cell = $ParticlesPerCell } | Out-Null

$fields = @('particle_count', 'granular_material_measured', 'granular_counted',
    'granular_yielded', 'granular_detached', 'granular_damaged', 'granular_invalid',
    'granular_min_plastic_jacobian', 'granular_mean_plastic_jacobian',
    'granular_max_plastic_jacobian', 'granular_hardening_capped', 'granular_strain_limited',
    'granular_solver_substeps', 'live_mean_speed_m_s', 'live_max_speed_m_s')
$r = Invoke-RtIpc fluid.run @{ steps = $Steps; sample_every = $SampleEvery; domains = @($Domain);
    fields = $fields; max_fields = @('granular_yielded', 'granular_detached',
    'granular_max_plastic_jacobian', 'granular_hardening_capped', 'granular_invalid') }

$rows = foreach ($s in $r.samples) {
    $x = $s.domains.$Domain
    [pscustomobject]@{
        step = $s.step; n = $x.particle_count; measured = $x.granular_material_measured
        counted = $x.granular_counted; yielded = $x.granular_yielded
        torn = $x.granular_detached; invalid = $x.granular_invalid
        jp_min = [math]::Round($x.granular_min_plastic_jacobian, 3)
        jp_mean = [math]::Round($x.granular_mean_plastic_jacobian, 3)
        jp_max = [math]::Round($x.granular_max_plastic_jacobian, 3)
        hcap = $x.granular_hardening_capped; clamp = $x.granular_strain_limited
        sub = $x.granular_solver_substeps
        v_mean = [math]::Round($x.live_mean_speed_m_s, 2)
        v_max = [math]::Round($x.live_max_speed_m_s, 2)
    }
}
$rows | Format-Table -AutoSize

$peak = $r.max.$Domain
$last = $rows[-1]
$verdict = @()
$verdict += if (@($rows | Where-Object { -not $_.measured }).Count -eq 0) { "PASS measured every sample" } else { "FAIL counters not measured" }
$verdict += if (@($rows | Where-Object { $_.counted -ne $_.n }).Count -eq 0) { "PASS counted == particles" } else { "FAIL counted != particles (mixed domain?)" }
$verdict += if ($peak.granular_yielded -gt 0) { "PASS yielded peak $($peak.granular_yielded)" } else { "FAIL never yielded" }
# The seed already spreads J_P by +-granular_snow_heterogeneity, so compaction
# and loosening are judged against the FIRST sample, not against 1.
$jp0min = $rows[0].jp_min; $jp0max = $rows[0].jp_max
$minJp = ($rows | Measure-Object jp_min -Minimum).Minimum
$verdict += "INFO initial J_P spread $jp0min .. $jp0max (heterogeneity $($d.granular_snow_heterogeneity))"
$verdict += if ($minJp -lt $jp0min - 0.01) { "PASS compaction J_P min $minJp" } else { "FAIL no compaction (J_P min $minJp)" }
$verdict += if ($peak.granular_max_plastic_jacobian -gt $jp0max + 0.01) { "PASS loosening J_P max $($peak.granular_max_plastic_jacobian)" } else { "FAIL never loosened past the seed spread: compacts but cannot crack" }
$floor = [math]::Round($d.granular_density / 917.0, 3)
$verdict += if ($minJp -ge $floor - 0.002) { "PASS J_P floor respected ($floor = rho/rho_ice)" } else { "FAIL J_P $minJp below ice density floor $floor" }
$sub = ($rows | Measure-Object sub -Maximum).Maximum
$verdict += "INFO max substeps $sub (ceiling $($d.granular_snow_max_hardening)x => about sqrt(ceiling) x the virgin count, constant)"
$verdict += if ($peak.granular_invalid -eq 0) { "PASS no invalid kinematics" } else { "FAIL invalid peak $($peak.granular_invalid)" }
$verdict += "INFO hardening clamp peak $($peak.granular_hardening_capped), last v_mean $($last.v_mean) m/s"
$verdict | ForEach-Object { Write-Host $_ }
