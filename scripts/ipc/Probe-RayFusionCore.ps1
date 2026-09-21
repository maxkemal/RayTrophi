<#
.SYNOPSIS
  Validate the compiled RayFusion probe control plane without changing a scene.
.NOTES
  User builds and starts the application first. This script neither compiles
  nor launches it. A passed core test is NOT a rendered GI or GPU test.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force
$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "[PASS] $Name" }
    else { Write-Host "[FAIL] $Name" -ForegroundColor Red; $script:failures++ }
}
try {
    Connect-RtIpc -TimeoutMs 3000
    $status = Invoke-RtIpc rayfusion.core_status @{}
    Check 'Core exists; renderer and GI are explicitly unavailable' (
        $status.core_available -eq $true -and $status.renderer_available -eq $false -and
        $status.gi_active -eq $false -and -not [string]::IsNullOrWhiteSpace($status.inactive_reason))
    Check 'Stage is probe_control_plane' ($status.stage -eq 'probe_control_plane')
    Check 'Directional payload ABI is internally consistent' (
        $status.probe_abi_version -eq 1 -and $status.directional_texels_per_probe -eq 64 -and
        $status.payload_bytes_per_probe -eq 32 * $status.directional_texels_per_probe)
    Check 'Planned work fits the ray budget' (
        $status.planned_rays_per_probe -gt 0 -and $status.planned_probes_per_update -gt 0 -and
        $status.planned_rays_per_probe * $status.planned_probes_per_update -le $status.planned_rays_per_update)
    $quality = Invoke-RtIpc viewport.quality @{}
    Check 'Budget reads the canonical viewport quality' ($quality.preset -eq $status.quality)

    $report = Invoke-RtIpc rayfusion.validate_core @{}
    Check 'Native fixtures ran and do not claim GPU coverage' (
        $report.checks.Count -ge 34 -and $report.gpu_tested -eq $false)
    foreach ($case in $report.checks) { Check ([string]$case.name) ($case.passed -eq $true) }
    Check 'Native aggregate passed' ($report.passed -eq $true)

    $rejected = $false
    try { Invoke-RtIpc rayfusion.validate_core @{ unexpected = 1 } | Out-Null }
    catch { $rejected = $true }
    Check 'Unknown parameters are rejected' $rejected
    if ($failures -gt 0) { throw "$failures RayFusion core checks failed" }
    Write-Host 'Core checks passed. GPU tracing, GI images and FPS were NOT tested.'
} finally {
    Disconnect-RtIpc
}
