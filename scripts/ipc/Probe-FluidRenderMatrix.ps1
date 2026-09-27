<#
.SYNOPSIS
Capture a liquid domain in every render mode x viewport shading and report the
volume tables next to each screenshot.

.DESCRIPTION
Faz 0 render matrix of docs/dev/BIRLESIK_MADDE_DOMAIN_TASARIMI.md. For each
render mode (particles / surface / fog) and each shading (solid / material /
rendered) it saves a viewport screenshot and prints the published volume count
per Vulkan backend plus the splat telemetry.

The domain must already hold particles. The viewport window must be visible:
the raster viewport only redraws when dirty, so the camera is nudged.

-ForceResync: before the 2026-09-27 fix, fluid.set_param render_mode did not
request a simulation render resync, so on a paused timeline Solid/Material drew
the PREVIOUS mode. This switch follows the mode change with fluid.set_fog
(which requests the resync) so an old exe still measures the right frame. On a
fixed exe, run WITHOUT it: a lagging raster column means the fix regressed.

.EXAMPLE
.\scripts\ipc\Probe-FluidRenderMatrix.ps1 -Domain Water -OutDir C:\temp\matrix
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Domain,
    [string[]]$Modes = @('particles', 'surface', 'fog'),
    [string[]]$Shadings = @('solid', 'material', 'rendered'),
    [Parameter(Mandatory)][string]$OutDir,
    [int]$SettleMs = 2500,
    [switch]$ForceResync
)

$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
New-Item -ItemType Directory -Force $OutDir | Out-Null

$before = Invoke-RtIpc fluid.get @{ domain = $Domain }
if ($before.particle_count -le 0) { throw "Domain '$Domain' has no particles." }
$shadingBefore = (Invoke-RtIpc viewport.shading).mode
$null = Invoke-RtIpc viewport.capture @{ enabled = $true }

$rows = @()
try {
    foreach ($mode in $Modes) {
        $null = Invoke-RtIpc fluid.set_param @{ domain = $Domain; render_mode = $mode }
        if ($ForceResync) {
            $null = Invoke-RtIpc fluid.set_fog @{ domain = $Domain; spread_voxels = [double]$before.fog_spread_voxels }
        }
        foreach ($sh in $Shadings) {
            $null = Invoke-RtIpc viewport.set_shading @{ mode = $sh }
            $null = Invoke-RtIpc camera.orbit @{ yaw = 0.5; pitch = 0.0 }
            Start-Sleep -Milliseconds $SettleMs
            $null = Invoke-RtIpc camera.orbit @{ yaw = -0.5; pitch = 0.0 }
            Start-Sleep -Milliseconds $SettleMs
            $file = Join-Path $OutDir ("{0}_{1}.jpg" -f $mode, $sh)
            $shot = $false
            $img = Invoke-RtIpc viewport.get_screenshot
            if ($img.image_base64) {
                [IO.File]::WriteAllBytes($file, [Convert]::FromBase64String($img.image_base64))
                $shot = $true
            }
            $info = Invoke-RtIpc fluid.get @{ domain = $Domain }
            $tables = Invoke-RtIpc render.volume_tables
            $tel = Invoke-RtIpc viewport.frame_telemetry
            $rows += [pscustomobject]@{
                mode      = $mode
                shading   = $sh
                reported  = $info.render_mode
                shot      = $shot
                volumes   = ($tables.backends | ForEach-Object { '{0}:{1}' -f $_.role, $_.instance_count }) -join ' '
                sphere_up = $tel.sphere_impostors_uploaded
                full_inst = $tel.full_instances
            }
        }
    }
} finally {
    $null = Invoke-RtIpc fluid.set_param @{ domain = $Domain; render_mode = $before.render_mode }
    if ($ForceResync) {
        $null = Invoke-RtIpc fluid.set_fog @{ domain = $Domain; spread_voxels = [double]$before.fog_spread_voxels }
    }
    $null = Invoke-RtIpc viewport.set_shading @{ mode = $shadingBefore }
}

$rows | Format-Table -AutoSize
Write-Host "Screenshots: $OutDir"
Write-Host ("Expected: on 'viewport', particles rows show ONE volume fewer than surface/fog " +
            "rows (other domains, e.g. gas, add to every row). Equal counts = the retired " +
            "liquid volume is still published.")
