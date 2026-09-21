param([int]$Frames=22, [switch]$BalancedRt)
$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
Add-Type @'
using System; using System.Runtime.InteropServices;
public class CutoutProbeFocus {
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
$path=Join-Path $PSScriptRoot ('../../tmp/cutout-live-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
$original=@(); $runs=@(); $success=$false; $restoreErrors=@()
$camera=$null; $status=$null
$quality=$null; $rtShadow=$null
$focus=[CutoutProbeFocus]::GetForegroundWindow()
function Drive([int]$count) {
    for($i=0;$i -lt $count;$i++) {
        $p=@([double]$camera.position[0],[double]$camera.position[1],[double]$camera.position[2])
        $p[0]+=($i%2)*0.0001
        Invoke-RtIpc camera.set_position @{position=$p} | Out-Null
    }
}
function Set-Cutout([int]$value) {
    foreach($m in $original) {
        Invoke-RtIpc material.set_param @{material_name=$m.name;param='alpha_cutout';value=$value} | Out-Null
        if((Invoke-RtIpc material.get_param @{material_name=$m.name;param='alpha_cutout'}) -ne $value) { throw "Cutout readback failed: $($m.name)" }
    }
}
function Measure-Cutout([string]$label) {
    Drive 10
    Invoke-RtIpc viewport.reset_frame_timings | Out-Null
    Drive $Frames
    $t=Invoke-RtIpc viewport.frame_timings
    if(!$t.available -or $t.frames_with_gpu -lt 10) { throw "Insufficient GPU frames: $label" }
    $script:runs += [pscustomobject]@{label=$label;timings=$t}
    [pscustomobject]@{camera=$camera;materials=$original;runs=$script:runs} | ConvertTo-Json -Depth 20 | Set-Content "$path.json" -Encoding UTF8
    [pscustomobject]@{label=$label;gpu=$t.frame_gpu_mean_ms;main=($t.stages|Where-Object name -eq main_pass).gpu_mean_ms;transmission=($t.stages|Where-Object name -eq transmission).gpu_mean_ms;depth=($t.stages|Where-Object name -eq depth_prepass).gpu_mean_ms;tris=$t.applied.visible_triangles;draws=$t.applied.draw_calls} | ConvertTo-Json -Compress
    # Capture outside the timing window and turn it off again before next arm.
    Invoke-RtIpc viewport.capture @{enabled=$true} | Out-Null
    Drive 3
    $shot=Invoke-RtIpc viewport.get_screenshot
    if(!$shot.image_base64) { throw "No screenshot: $label" }
    [IO.File]::WriteAllBytes("$path-$label.jpg",[Convert]::FromBase64String($shot.image_base64))
    Invoke-RtIpc viewport.capture @{enabled=$false} | Out-Null
}
try {
    $camera=Invoke-RtIpc camera.get
    $status=Invoke-RtIpc viewport.status
    if($BalancedRt) {
        $quality=Invoke-RtIpc viewport.quality
        $rtShadow=Invoke-RtIpc viewport.rt_shadow
    }
    $mats=@(Invoke-RtIpc material.list | Where-Object { $_.type -eq 'principled' -and $_.name -like 'foliage_*' })
    foreach($m in $mats) {
        $textures=@(Invoke-RtIpc material.textures @{material_name=$m.name})
        if($textures.slot -contains 'opacity') {
            $old=Invoke-RtIpc material.get_param @{material_name=$m.name;param='alpha_cutout'}
            $tr=Invoke-RtIpc material.get_param @{material_name=$m.name;param='transmission'}
            if($tr -gt 0.001 -or $textures.slot -contains 'transmission') { throw "Explicit transmission on candidate: $($m.name)" }
            $original += [pscustomobject]@{name=$m.name;alpha_cutout=$old;transmission=$tr;textures=$textures}
        }
    }
    if(!$original.Count) { throw 'No live foliage materials with opacity maps found' }
    Write-Output "MATERIALS=$($original.Count)"
    $original | Select-Object name,alpha_cutout | ConvertTo-Json -Compress
    [pscustomobject]@{camera=$camera;materials=$original} | ConvertTo-Json -Depth 12 | Set-Content "$path-original.json" -Encoding UTF8
    [void][CutoutProbeFocus]::SetForegroundWindow((Get-Process RayTrophiStudio).MainWindowHandle)
    if($BalancedRt) {
        Invoke-RtIpc viewport.set_quality @{preset='balanced'} | Out-Null
        Invoke-RtIpc viewport.set_rt_shadow @{enabled=$true} | Out-Null
    }
    Invoke-RtIpc viewport.capture @{enabled=$false} | Out-Null
    Set-Cutout 0; Measure-Cutout 'off'
    Set-Cutout 1; Measure-Cutout 'on'
    Set-Cutout 0; Measure-Cutout 'off_repeat'
    Set-Cutout 1; Measure-Cutout 'on_repeat'
    $success=$true
    Write-Output "RESULT=$path.json"
} finally {
    if(!$success) {
        foreach($m in $original) {
            try { Invoke-RtIpc material.set_param @{material_name=$m.name;param='alpha_cutout';value=$m.alpha_cutout} | Out-Null } catch { $restoreErrors+=$_.ToString() }
        }
    }
    if($camera) {
        try { Invoke-RtIpc camera.set_position @{position=@($camera.position)} | Out-Null; Invoke-RtIpc camera.set_target @{target=@($camera.target)} | Out-Null } catch { $restoreErrors+=$_.ToString() }
    }
    if($status) { try { Invoke-RtIpc viewport.capture @{enabled=[bool]$status.capture_enabled} | Out-Null } catch { $restoreErrors+=$_.ToString() } }
    if($quality) { try { Invoke-RtIpc viewport.set_quality @{preset=$quality.preset} | Out-Null } catch { $restoreErrors+=$_.ToString() } }
    if($rtShadow) { try { Invoke-RtIpc viewport.set_rt_shadow @{enabled=[bool]$rtShadow.enabled} | Out-Null } catch { $restoreErrors+=$_.ToString() } }
    [void][CutoutProbeFocus]::SetForegroundWindow($focus)
    Disconnect-RtIpc
    if($restoreErrors.Count) { throw ($restoreErrors -join "`n") }
    Write-Output "CAMERA_RESTORED; CUTOUT_LEFT_ON=$success"
}
