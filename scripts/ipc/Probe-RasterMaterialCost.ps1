param([int]$Frames = 18, [switch]$OpacityOnly, [switch]$TransmissionOnly, [switch]$BaselineOnly)
$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
Add-Type @'
using System; using System.Runtime.InteropServices;
public class RasterProbeFocus {
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
$saved = if (!$BaselineOnly) { Get-Content "$PSScriptRoot/../../tmp/raster-material-snapshot.json" -Raw | ConvertFrom-Json } else { @() }
$runs = @()
$changed = @()
$restoreErrors = @()
$changedParam = if ($OpacityOnly) { 'opacity' } else { 'transmission' }
$resultName = if ($OpacityOnly) { 'raster-opacity-cost.json' } elseif ($TransmissionOnly) { 'raster-transmission-cost.json' } else { 'raster-material-cost.json' }
if ($BaselineOnly) { $resultName = 'raster-postbuild-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json' }
$world = Invoke-RtIpc world.get
$shading = Invoke-RtIpc viewport.shading
$camera = Invoke-RtIpc camera.get
$focus = [RasterProbeFocus]::GetForegroundWindow()
$app = Get-Process RayTrophiStudio -ErrorAction Stop
[void][RasterProbeFocus]::SetForegroundWindow($app.MainWindowHandle)
function Drive([int]$Count) {
    for ($i=0; $i -lt $Count; $i++) {
        $p = @([double]$camera.position[0], [double]$camera.position[1], [double]$camera.position[2])
        $p[0] += ($i % 2) * 0.0001
        Invoke-RtIpc camera.set_position @{position=$p} | Out-Null
    }
}
function Measure-Cost([string]$Label) {
    Drive 8
    Invoke-RtIpc viewport.reset_frame_timings | Out-Null
    Drive $Frames
    $t = Invoke-RtIpc viewport.frame_timings
    if (!$t.available -or $t.frames_with_gpu -lt 5) { throw "Insufficient GPU samples: $Label" }
    $script:runs += [pscustomobject]@{label=$Label;timings=$t}
    $script:runs | ConvertTo-Json -Depth 20 | Set-Content "$PSScriptRoot/../../tmp/$resultName" -Encoding UTF8
    [pscustomobject]@{label=$Label;tris=$t.applied.visible_triangles;draws=$t.applied.draw_calls;gpu=$t.frame_gpu_mean_ms;main=($t.stages | Where-Object name -eq main_pass).gpu_mean_ms;transmission=($t.stages | Where-Object name -eq transmission).gpu_mean_ms} | ConvertTo-Json -Compress
}
try {
    Measure-Cost 'baseline'
    if ($BaselineOnly) {
        Measure-Cost 'baseline_repeat'
        Write-Output "RESULT_FILE=$resultName"
    } elseif ($OpacityOnly) {
        foreach ($m in $saved) {
            if ($m.textures.slot -contains 'opacity') {
                $changed += $m
                Invoke-RtIpc material.set_param @{material_name=$m.name;param='opacity';value=0.0} | Out-Null
                if ((Invoke-RtIpc material.get_param @{material_name=$m.name;param='opacity'}) -ne 0) { throw "Opacity write not applied: $($m.name)" }
            }
        }
        Measure-Cost 'foliage_opacity_zero'
        foreach ($m in $changed) {
            Invoke-RtIpc material.set_param @{material_name=$m.name;param='opacity';value=[double]$m.opacity} | Out-Null
        }
        $changed = @()
        Measure-Cost 'baseline_after_opacity'
    } else {
    if (!$TransmissionOnly) {
    Invoke-RtIpc world.set_mode @{mode='solid'} | Out-Null
    Measure-Cost 'world_solid'
    Invoke-RtIpc world.set_mode @{mode=$world.mode} | Out-Null
    Measure-Cost 'baseline_after_sky'
    }
    foreach ($m in $saved) {
        if ($m.transmission -gt 0) {
            $changed += $m
            Invoke-RtIpc material.set_param @{material_name=$m.name;param='transmission';value=0.0} | Out-Null
            if ((Invoke-RtIpc material.get_param @{material_name=$m.name;param='transmission'}) -ne 0) { throw "Transmission write not applied: $($m.name)" }
        }
    }
    Measure-Cost 'explicit_transmission_zero'
    foreach ($m in $changed) {
        Invoke-RtIpc material.set_param @{material_name=$m.name;param='transmission';value=[double]$m.transmission} | Out-Null
    }
    $changed = @()
    Measure-Cost 'baseline_after_transmission'
    if (!$TransmissionOnly) {
    Invoke-RtIpc viewport.set_shading @{mode='solid'} | Out-Null
    Measure-Cost 'solid_shading'
    }
    }
} finally {
    foreach ($m in $changed) {
        try { Invoke-RtIpc material.set_param @{material_name=$m.name;param=$changedParam;value=[double]$m.$changedParam} | Out-Null } catch { $restoreErrors += $_.ToString() }
    }
    if (!$BaselineOnly) {
        try { Invoke-RtIpc world.set_mode @{mode=$world.mode} | Out-Null } catch { $restoreErrors += $_.ToString() }
        try { Invoke-RtIpc viewport.set_shading @{mode=$shading.mode;matcap_preset=$shading.matcap_preset} | Out-Null } catch { $restoreErrors += $_.ToString() }
    }
    try { Invoke-RtIpc camera.set_position @{position=@($camera.position)} | Out-Null; Invoke-RtIpc camera.set_target @{target=@($camera.target)} | Out-Null } catch { $restoreErrors += $_.ToString() }
    [void][RasterProbeFocus]::SetForegroundWindow($focus)
    Disconnect-RtIpc
    if ($restoreErrors.Count) { throw ($restoreErrors -join "`n") }
    Write-Output 'RESTORE_OK'
}
