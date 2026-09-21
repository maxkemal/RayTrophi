param([ValidateRange(12,240)][int]$Frames=60)
$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
$folder=Join-Path $PSScriptRoot ('../../tmp/screen-gi-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$folder=(Resolve-Path -LiteralPath $folder).Path
$saved=$null; $camera=$null; $capture=$false; $haveCapture=$false; $runs=@(); $restoreErrors=@()
function Drive([int]$count) {
    for ($i=0;$i -lt $count;$i++) {
        Invoke-RtIpc camera.set_position @{position=@([double]$camera.position[0],[double]$camera.position[1],[double]$camera.position[2])} | Out-Null
    }
}
function Set-Gi([bool]$enabled,[int]$samples) {
    Invoke-RtIpc rayfusion.set_screen_gi @{enabled=$enabled;samples=$samples;filter_radius=[int]$saved.filter_radius;max_distance=[double]$saved.max_distance} | Out-Null
}
try {
    $saved=Invoke-RtIpc rayfusion.screen_gi
    $camera=Invoke-RtIpc camera.get
    $status=Invoke-RtIpc viewport.status
    $capture=[bool]$status.capture_enabled; $haveCapture=$true
    if ((Invoke-RtIpc viewport.shading).mode -ne 'material') {throw 'Requires material viewport; mode was not changed.'}
    if (!(Invoke-RtIpc viewport.rt_shadow).ready) {throw 'Requires active RT shadows; shadow settings were not changed.'}
    $initial=[ordered]@{screen_gi=$saved;camera=$camera;viewport=$status;probe=(Invoke-RtIpc rayfusion.probe_field);light=(Invoke-RtIpc lights.get @{index=0});materials=(Invoke-RtIpc material.list)}
    $initial | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath "$folder/initial.json" -Encoding UTF8
    $appProcess=Get-Process RayTrophiStudio | Select-Object -First 1
    $focus=New-Object -ComObject WScript.Shell
    [void]$focus.AppActivate($appProcess.Id)
    foreach ($arm in @(@{name='off';enabled=$false;samples=1},@{name='one';enabled=$true;samples=1},@{name='four';enabled=$true;samples=4},@{name='off_repeat';enabled=$false;samples=1})) {
        Set-Gi $arm.enabled $arm.samples
        Invoke-RtIpc viewport.capture @{enabled=$false} | Out-Null
        Drive 8
        Invoke-RtIpc viewport.reset_frame_timings | Out-Null
        Drive $Frames
        $timing=Invoke-RtIpc viewport.frame_timings
        if (!$timing.available -or $timing.frames_with_gpu -lt 8) {throw "Insufficient raster GPU frames for $($arm.name); keep application visible."}
        if ($timing.applied.screen_gi.enabled -ne $arm.enabled -or ($arm.enabled -and !$timing.applied.screen_gi.ready)) {throw "GI arm not applied: $($arm.name)"}
        if ($runs.Count -gt 0 -and ($timing.applied.visible_triangles -ne $runs[0].timings.applied.visible_triangles -or $timing.applied.width -ne $runs[0].timings.applied.width -or $timing.applied.height -ne $runs[0].timings.applied.height)) {throw 'Geometry or resolution changed between arms.'}
        $runs+=[pscustomobject]@{name=$arm.name;timings=$timing}
        $runs | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath "$folder/timings.json" -Encoding UTF8
        $stages=@{};foreach ($stage in $timing.stages) {$stages[$stage.name]=$stage.gpu_mean_ms}
        [pscustomobject]@{arm=$arm.name;gpu_ms=$timing.frame_gpu_mean_ms;main_ms=$stages.main_pass;gi_trace_ms=$stages.screen_gi_trace;gi_filter_ms=$stages.screen_gi_filter;triangles=$timing.applied.visible_triangles;frames=$timing.frames;folder=$folder} | ConvertTo-Json -Compress
        Invoke-RtIpc viewport.capture @{enabled=$true} | Out-Null
        Drive 4
        $shot=Invoke-RtIpc viewport.get_screenshot
        if (!$shot.image_base64) {throw 'No screenshot returned'}
        [IO.File]::WriteAllBytes("$folder/$($arm.name).jpg",[Convert]::FromBase64String($shot.image_base64))
    }
} finally {
    if ($saved) {try {Set-Gi ([bool]$saved.enabled) ([int]$saved.samples)} catch {$restoreErrors+=$_.ToString()}}
    if ($haveCapture) {try {Invoke-RtIpc viewport.capture @{enabled=$capture} | Out-Null} catch {$restoreErrors+=$_.ToString()}}
    if ($camera) {try {Drive 2} catch {$restoreErrors+=$_.ToString()}}
    [pscustomobject]@{restore_errors=$restoreErrors;saved=$saved} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$folder/restore.json" -Encoding UTF8
    Disconnect-RtIpc
    if ($restoreErrors.Count) {Write-Warning ($restoreErrors -join '; ')}
}
